import copy,math,unittest
import cost_model as m

class CostModelTests(unittest.TestCase):
    def test_solver_known_coefficients(self):
        X=[[1,float(i),float(j)] for i in range(1,7) for j in [1,3,8]];y=[2+3*r[1]+.5*r[2] for r in X]
        fit=m.fit_nnls(X,y,ridge=0)
        for got,want in zip(fit['coefficients'],[2,3,.5]):self.assertAlmostEqual(got,want,places=4)
        self.assertEqual(fit['normalized_design_rank'],3)

    def test_positive_scale_invariance(self):
        X=[[1,i,i*i] for i in range(1,10)];y=[7+2*i+.3*i*i for i in range(1,10)]
        a=m.fit_nnls(X,y);Z=[[r[0],r[1]*1e6,r[2]*1e-5] for r in X];b=m.fit_nnls(Z,y)
        for x,z in zip(X,Z):self.assertAlmostEqual(m.value(a,x),m.value(b,z),places=5)

    def test_collinear_and_zero_columns(self):
        X=[[1,i,2*i,0] for i in range(1,10)];fit=m.fit_nnls(X,[5+3*i for i in range(1,10)])
        self.assertEqual(fit['normalized_design_rank'],2);self.assertEqual(fit['coefficients'][3],0)
        self.assertTrue(all(x>=0 for x in fit['coefficients']))

    def test_invalid_measurement(self):
        for bad in [float('nan'),float('inf'),0,-1]:
            with self.assertRaises(ValueError):m.fit_nnls([[1,1],[1,2]],[1,bad])

    def case(self,T=64):
        return {'quant':'mxfp4','direction':'up','K':5120,'M':2304,'experts':6,'tokens':T,'topk':6,'expert_bytes':6266880,'histogram':[T]*6}

    def model(self):
        c=self.case();cases=[self.case(64),self.case(128)];targets={}
        for t in m.TARGETS:
            names,x=m.features(c,t);X=[m.features(q,t)[1] for q in cases];scales=[max(1,abs(v)) for v in X[1]]
            targets[t]={'features':names,'coefficients':[20 if t.startswith('cpu') else 5]+[0]*(len(x)-1),
                'scales':scales,'feature_support':m.feature_support(X,scales),'supported_calibration_count':6,
                'error_envelope':{'lower_divisor':1.1,'upper_multiplier':1.1,'timing_MAD_floor_ms':.1}}
        signature={'scope':'GPU activations and host routes ready -> GPU output ready'}
        model={'schema':m.VERSION,'implementation_sha256':m.IMPLEMENTATION_SHA256,'timing_signature':signature,
            'timing_signature_sha256':m.signature_hash(signature),'split_forecasts':'unsupported',
            'strata':{m.key(c):{'support':{'declared_experts':[6],'tokens_min':64,'tokens_max':128,'active_min':6,'active_max':6,'max_expert_rows':128},'targets':targets}}}
        model['model_id']=m.digest(model);return model

    def event(self):
        model=self.model()
        return {'current_assignment':'cpu','timing_signature_sha256':model['timing_signature_sha256'],'feature_stage':'host_routes_ready',
            'decision_stage':'post_assignment_shadow','unique_routes_validated':True,'native_moe_cache':False,'host_weights_ready':True,
            'gpu_weight_resident':False,'cpu_queue_ms':0,'gpu_queue_ms':0,'operator':self.case()}

    def test_shadow_never_applies(self):
        result=m.advise(self.model(),self.event());self.assertEqual(result['hypothetical_choice'],'gpu')
        self.assertEqual(result['assignment_after'],'cpu');self.assertFalse(result['would_change_assignment'])

    def test_error_band_overlap_holds_assignment(self):
        model=self.model()
        for r in model['strata'][m.key(self.case())]['targets'].values():r['error_envelope']['lower_divisor']=10;r['error_envelope']['upper_multiplier']=10
        model.pop('model_id');model['model_id']=m.digest(model)
        result=m.advise(model,self.event());self.assertIn('empirical_error_envelopes_overlap',result['abstain_reasons'])

    def test_missing_provenance_and_queue_fail_closed(self):
        for field,value in [('timing_signature_sha256','other'),('host_weights_ready',False),('native_moe_cache',True),('gpu_queue_ms',None),('cpu_queue_ms',1),('decision_stage','pre_assignment'),('unique_routes_validated',False)]:
            event=self.event();event[field]=value;r=m.advise(self.model(),event)
            self.assertIsNone(r['hypothetical_choice']);self.assertTrue(r['abstain_reasons']);self.assertEqual(r['predictions'],{})

    def test_resident_requires_host_weights(self):
        event=self.event();event['gpu_weight_resident']=True;event['host_weights_ready']=False
        self.assertIn('CPU_host_weights_not_verified_available',m.advise(self.model(),event)['abstain_reasons'])

    def test_resident_uses_separate_measured_target(self):
        model=self.model();event=self.event();event['gpu_weight_resident']=True
        target=model['strata'][m.key(self.case())]['targets']['gpu_resident_ms'];target['coefficients'][0]=3
        model.pop('model_id');model['model_id']=m.digest(model)
        result=m.advise(model,event)
        self.assertEqual(result['predictions']['gpu']['service_ms'],3)
        self.assertEqual(result['hypothetical_choice'],'gpu')

    def test_unmeasured_384_experts(self):
        event=self.event();event['operator']['experts']=384;event['operator']['histogram']+=[0]*378
        r=m.advise(self.model(),event);self.assertIn('unmeasured_declared_expert_count',r['abstain_reasons'])

    def test_invalid_histograms(self):
        for h in [[64]*5,[65]*6,[-1,65,64,64,64,64],[63]*6]:
            c=self.case();c['histogram']=h
            with self.assertRaises(ValueError):m.validate_case(c)

    def test_transfer_runs_padding(self):
        c=self.case(64);c.update(experts=8,topk=2,histogram=[64,0,0,0,0,0,0,64])
        v=m.validate_case(c);self.assertEqual(v['transfer_runs'],2);self.assertEqual(v['submitted_weight_bytes'],2*c['expert_bytes']+512)

    def test_model_tamper_rejected(self):
        model=self.model();model['strata'][m.key(self.case())]['targets']['gpu_stream_ms']['coefficients'][0]=1
        with self.assertRaises(ValueError):m.advise(model,self.event())

    def test_feature_extrapolation_rejected(self):
        model=self.model();reg=model['strata'][m.key(self.case())]['targets']['gpu_stream_ms'];x=m.features(self.case(1024),'gpu_stream_ms')[1]
        self.assertEqual(m.check_feature_support(reg,x),'feature_range_extrapolation')

    def test_evidence_plan_completeness(self):
        plan={'cases':['a'],'workloads':[{'id':'test-0','split':'test','sha256':'x'}]}
        with self.assertRaises(ValueError):m.expected_records(plan,[],{'test'})
        row={'case':{'name':'a'},'workload':{'id':'test-0','split':'test','sha256':'x'}}
        m.expected_records(plan,[row],{'test'})
        with self.assertRaises(ValueError):m.expected_records(plan,[row,row],{'test'})

if __name__=='__main__':unittest.main()
