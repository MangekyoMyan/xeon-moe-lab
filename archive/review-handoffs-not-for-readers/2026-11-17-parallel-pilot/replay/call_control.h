#pragma once
#include "route37.h"
struct CallResult {
    bool label_accepted=false,decode_returned=false,output_called=false,output_valid=false;
    bool synchronized=false,drain_called=false,finalize_called=false,finalize_ok=false;
    int decode_status=-10000,exception_stage=0;
    bool ok()const{return label_accepted&&decode_returned&&decode_status==0&&output_valid&&synchronized&&finalize_ok;}
};
// No allocation in the coordinator. Completion is called once, after either
// normal output synchronization or error drain. No-eval failures are preserved.
// -10000 means a controlling-caller exception with no public decode return;
// the actual public status, if returned, is never replaced.
template<class Api,class Submit,class Output,class Drain>
CallResult drive_call(Api &api,const void *owner,const xeon_route37_label &tag,
                     Submit submit,Output output,Drain drain){
    CallResult r;r.label_accepted=api.label(owner,&tag)!=0;if(!r.label_accepted)return r;
    try{r.decode_status=submit();r.decode_returned=true;}catch(...){r.exception_stage=1;}
    if(r.decode_returned&&r.decode_status==0){
        r.output_called=true;
        try{r.output_valid=output();r.synchronized=true;}catch(...){r.exception_stage=2;}
    }
    if(!r.synchronized){
        r.drain_called=true;
        try{drain();r.synchronized=true;}catch(...){r.exception_stage=3;}
    }
    r.finalize_called=true;r.finalize_ok=api.finish(owner,r.decode_status,r.synchronized)!=0;
    return r;
}
