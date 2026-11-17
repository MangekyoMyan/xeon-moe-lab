#!/usr/bin/env python3
"""Create original, untokenized pilot fixtures. No model/GPU/API or network."""
from pathlib import Path
import hashlib
import json

OUT = Path(__file__).resolve().parent
families = []


def variant(case_id, lane, prompt, reference, *, code=None, language=None):
    result = dict(case_id=case_id, language_or_content=lane,
                  prompt_utf8=prompt.strip()+'\n', reference_utf8=reference.strip()+'\n',
                  reference_role='fixed teacher-forced continuation; not a sampled model answer',
                  capture_t1_calls=64, prefix_policy='full rendered prompt; record actual prefix length',
                  reference_window_policy='first 64 continuation token IDs after a verified assistant boundary',
                  tokenization_performed=False, rendered_prefix_sha256=None,
                  prefix_token_ids=None, continuation_token_ids=None,
                  tokenizer_identity=None, chat_template_identity=None)
    if code is not None:
        result['reference_code_utf8'] = code.strip()+'\n'
        result['reference_code_language'] = language
        result['reference_starts_with_code'] = True
    return result


def family(fid, kind, split, facts, variants):
    families.append(dict(family_id=fid, kind=kind, split=split,
                         split_frozen_before_routes=True,
                         translations_ports_windows_turns_must_remain_in_this_split=True,
                         original_synthetic_material=True, common_semantic_contract=facts,
                         variants=variants))


family('text_cache_budget','parallel_text','train',
       dict(budget_MiB=40,sizes_MiB=dict(A=24,B=16,C=16,D=8,E=8),
            requests=[['A','B'],['C','D'],['A','E'],['B','C'],['D','E'],['A','C']],
            candidates=dict(X=['A','B'],Y=['B','C','D'],Z=['A','D','E']),
            expected_sizes_MiB=dict(X=40,Y=40,Z=40),expected_hits=dict(X=5,Y=7,Z=7),
            expected_misses=dict(X=7,Y=5,Z=5),best_candidates_by_hits=['Y','Z']),[
variant('text_cache_budget_ja','ja',r'''
市の資料室で、地図データを読み出すための小さな静的キャッシュを検討しています。これは架空の計算例です。容量は40 MiBで、データAは24 MiB、BとCはそれぞれ16 MiB、DとEはそれぞれ8 MiBです。読み出し要求は順に {A,B}、{C,D}、{A,E}、{B,C}、{D,E}、{A,C} の6件でした。各組に含まれる各データを1回の参照として数え、キャッシュに置いたデータへの参照を1 hitとします。途中の入れ替えはありません。
候補XはAとB、候補YはBとCとD、候補ZはAとDとEを置きます。三つの候補の使用容量、6要求を通じたhit数とmiss数を計算してください。hit数が最大の候補が一つとは限らない場合、その点を明確にしてください。最後に、この参照回数だけでは利用者の待ち時間を何秒減らせるかを決められない理由を説明してください。データの内容を変えたり、要求を間引いたりしてはいけません。
''',r'''
三つの候補はすべて40 MiBに収まります。Xは24+16=40 MiB、Yは16+16+8=40 MiB、Zは24+8+8=40 MiBです。要求は各2データの6件なので、参照は合計12回あります。
Aは3回、Bは2回、Cは3回、Dは2回、Eは2回参照されています。したがってXのhit数は3+2=5で、miss数は12-5=7です。Yのhit数は2+3+2=7、miss数は5です。Zも3+2+2=7 hit、5 missとなります。hit数だけを基準にするなら、YとZが同率で最良です。どちらか一方を唯一の最適解とは言えません。
この計算は参照回数の比較です。実際の待ち時間を求めるには、データごとの読出し時間、要求内で並行に読めるか、転送やキャッシュ参照の固定費などが必要です。また、同じ合計miss数でも、個々の要求にmissが集中する程度は変わります。ここから特定の秒数や速度倍率を推定することはできません。まずこの6件に対する回数の結果を保持し、時間の測定を別に加えるのが適切です。
'''),
variant('text_cache_budget_en','en',r'''
A municipal archive is considering a small static cache for reading map data. This is a fictional calculation. The capacity is 40 MiB. Data item A occupies 24 MiB, B and C occupy 16 MiB each, and D and E occupy 8 MiB each. Six requests arrived in this order: {A,B}, {C,D}, {A,E}, {B,C}, {D,E}, and {A,C}. Count each item in each request as one reference. A reference to an item in the cache is one hit. There are no replacements during the sequence.
Candidate X stores A and B. Candidate Y stores B, C, and D. Candidate Z stores A, D, and E. Calculate the occupied capacity, total hits, and total misses for each candidate. Make clear if more than one candidate has the maximum number of hits. Finally, explain why these reference counts alone cannot determine how many seconds of user waiting time would be saved. Do not change the data contents or discard requests.
''',r'''
All three candidates occupy exactly 40 MiB. X uses 24+16=40 MiB, Y uses 16+16+8=40 MiB, and Z uses 24+8+8=40 MiB. Six requests with two items each produce twelve references in total.
A is referenced three times, B twice, C three times, D twice, and E twice. Candidate X therefore has 3+2=5 hits and 12-5=7 misses. Candidate Y has 2+3+2=7 hits and five misses. Candidate Z also has 3+2+2=7 hits and five misses. Under an objective that counts hits only, Y and Z tie for the best result. Neither is the unique optimum on this evidence.
These calculations compare reference counts. Predicting waiting time would also require the read time of each item, whether the items in one request can be read concurrently, and the fixed costs of transfers and cache access. Equal total miss counts can also be distributed differently across individual requests. The counts do not establish a particular number of seconds saved or a speed ratio. Preserve the result for these six requests and add direct timing as a separate measurement.
''')])

family('text_unit_conversion','parallel_text','train',
       dict(interval_minutes=[10,5,8],flow_L_per_min=[1.2,.8,1.5],
            indicated_total_L=27.6,expected_total_L=28,expected_minutes=23,
            expected_mL=28000,expected_seconds=1380,
            expected_average_L_per_min_fraction=[28,23],
            expected_average_mL_per_s_fraction=[28000,1380],
            expected_shortfall_percent_fraction=[10,7]),[
variant('text_unit_conversion_ja','ja',r'''
試験用ポンプの運転記録を単位付きで整理してください。最初の10分間は毎分1.2 L、次の5分間は毎分0.8 L、その次の8分間は毎分1.5 Lの流量を設定しました。各区間の流量は区間内で一定だったと仮定します。区間の間に休止時間はありません。終了時の積算計は27.6 Lを示しました。この記録と数値は説明のための架空のものです。
設定流量から計算した区間別の量と合計量、運転時間全体に対する平均流量を示してください。合計量をmL、合計時間を秒に直し、平均流量をmL/sでも示してください。最後に、設定から計算した合計量を分母として、積算計の表示が何パーセント少ないかを求めてください。表示の差だけでポンプの故障や積算計の誤りを断定せず、計算上確認できることと、追加測定が必要なことを区別してください。
''',r'''
各区間の量は「流量×時間」で求めます。最初は1.2×10=12 L、次は0.8×5=4 L、最後は1.5×8=12 Lです。設定から計算した合計量は12+4+12=28 L、合計時間は10+5+8=23分となります。
全体の平均流量は28/23 L/minで、約1.2174 L/minです。三つの流量を単純平均するのではなく、それぞれの継続時間を反映した値を使います。合計28 Lは28,000 mL、23分は1,380秒なので、平均は28,000/1,380で約20.2899 mL/sです。
積算計の27.6 Lは、計算値28 Lより0.4 L少なくなっています。指定された分母での差は0.4/28×100で、約1.4286%です。この結果は設定値と表示値の差を表します。実際の流量が設定どおりだったか、時間の記録が正確だったか、計器の校正や読取りに差がないかは、この表だけでは確認できません。故障の断定には、それらを区別できる追加の測定が必要です。
'''),
variant('text_unit_conversion_en','en',r'''
Organize this test-pump record with explicit units. The pump was set to 1.2 L per minute for the first ten minutes, 0.8 L per minute for the next five minutes, and 1.5 L per minute for the following eight minutes. Assume that each interval had a constant flow and that there were no pauses between intervals. At the end, the totalizing meter indicated 27.6 L. The record and its numbers are fictional examples.
Calculate the amount for each interval, the total amount implied by the settings, and the average flow over the complete running time. Convert the total amount to mL and the total duration to seconds, and report the average in mL/s as well. Finally, calculate the percentage by which the meter indication is lower, using the calculated total amount as the denominator. Do not conclude from the discrepancy alone that either the pump or the meter is faulty. Separate what the arithmetic establishes from what would need additional measurements.
''',r'''
The amount in each interval is flow multiplied by duration. The first interval gives 1.2×10=12 L, the second gives 0.8×5=4 L, and the last gives 1.5×8=12 L. The total implied by the settings is 12+4+12=28 L, over 10+5+8=23 minutes.
The average flow over the entire run is 28/23 L/min, approximately 1.2174 L/min. It is duration-weighted, rather than the unweighted average of the three settings. The total of 28 L equals 28,000 mL, and 23 minutes equals 1,380 seconds. The average is therefore 28,000/1,380, approximately 20.2899 mL/s.
The meter indication of 27.6 L is 0.4 L below the calculated 28 L. With the specified denominator, the discrepancy is 0.4/28×100, approximately 1.4286%. This establishes a difference between the settings-based calculation and the meter indication. The record does not establish whether actual flow followed the settings, whether the timing was exact, or whether calibration or reading differences affected the instruments. Additional measurements are needed to distinguish those possibilities before attributing the discrepancy to a fault.
''')])

family('text_plan_constraints','parallel_text','validation',
       dict(start='09:00',all_available_until='11:00',
            tasks=dict(S=dict(minutes=20,people=['Aya'],after=[]),
                       I=dict(minutes=30,people=['Aya','Ben'],after=['S']),
                       E=dict(minutes=40,people=['Ben','Chen'],after=['I']),
                       R=dict(minutes=25,people=['Aya'],after=['I']),
                       F=dict(minutes=15,people=['Aya','Ben','Chen'],after=['E','R'])),
            expected_schedule=dict(S=['09:00','09:20'],I=['09:20','09:50'],E=['09:50','10:30'],R=['09:50','10:15'],F=['10:30','10:45']),
            earliest_finish='10:45',critical_path=['S','I','E','F']),[
variant('text_plan_constraints_ja','ja',r'''
Aya、Ben、Chenの3人が、架空の展示装置を点検します。全員が09:00から11:00まで作業可能です。作業Sは準備で20分、Ayaが担当します。Sの後に行う作業Iは点検で30分、AyaとBenの両方が必要です。Iの後には、BenとChenが40分行う作業Eと、Ayaが25分行う記録作業Rがあります。最後の作業Fは15分で、EとRの両方が終了してから3人全員で行います。
一人は同じ時刻に二つの作業をできません。作業を途中で分割したり、担当者を変更したり、所要時間を短縮したりすることはできません。これ以外に部屋や装置の共用制約はなく、異なる担当者の作業は並行できます。終了時刻が最も早くなる実行可能な予定を作り、各作業の開始と終了を記してください。その時刻より早く完了できない理由も、先行関係に基づいて説明してください。
''',r'''
最も早い終了時刻は10:45です。Sを09:00〜09:20、Iを09:20〜09:50に行います。その後、EをBenとChenで09:50〜10:30、RをAyaが09:50〜10:15に行います。EとRは担当者が重ならないので並行できます。
Fは両方の終了を待つ必要があり、早くても10:30からです。3人が10:30〜10:45にFを行えば、全作業が完了します。AyaはR終了後の10:15〜10:30に待ち時間がありますが、Rを早く終えてもEの終了前にFを始めることはできません。
これより早い完了が不可能なのは、S→I→E→Fの先行関係があるためです。この経路の所要時間は20+30+40+15=105分で、09:00から連続して実行しても10:45になります。提示した予定はこの下限を達成し、担当者の重複や作業の分割もありません。したがって、与えられた制約の範囲で最も早い予定です。
'''),
variant('text_plan_constraints_en','en',r'''
Aya, Ben, and Chen will inspect a fictional exhibition device. All three are available from 09:00 to 11:00. Preparation task S takes twenty minutes and requires Aya. Inspection task I follows S, takes thirty minutes, and requires both Aya and Ben. After I, task E requires Ben and Chen for forty minutes, while documentation task R requires Aya for twenty-five minutes. Final task F takes fifteen minutes, requires all three people, and can begin only after both E and R finish.
No person can perform two tasks at once. Tasks cannot be split, their assigned people cannot be changed, and their durations cannot be shortened. There are no additional room or equipment constraints, so tasks with different assigned people may run concurrently. Give a feasible schedule with the earliest possible completion, including the start and finish of every task. Explain from the precedence constraints why completion cannot be earlier.
''',r'''
The earliest completion is 10:45. Run S from 09:00 to 09:20 and I from 09:20 to 09:50. Then Ben and Chen perform E from 09:50 to 10:30, while Aya performs R from 09:50 to 10:15. E and R can run concurrently because their assigned people do not overlap.
F must wait for both tasks, so its earliest start is 10:30. All three perform F from 10:30 to 10:45. Aya has an idle interval from 10:15 to 10:30 after finishing R, but finishing R earlier cannot allow F to begin before E is complete.
The precedence chain S→I→E→F requires 20+30+40+15=105 minutes. Even with no idle time on that chain, a 09:00 start cannot finish before 10:45. The proposed schedule reaches this lower bound while respecting each person's availability, avoiding overlapping assignments, and leaving all tasks uninterrupted. It is therefore an earliest schedule under the stated constraints.
''')])

family('text_document_summary','parallel_text','holdout',
       dict(sessions=6,seats_per_session=12,distinct_requests=90,reservations=72,
            attendees=64,no_shows=8,unreserved_requests=18,survey_responses=44,
            survey_good=38,survey_neutral=6,staff_extra_minutes_per_session=30,
            proposed_measures=['reminder','opt-in waiting list'],measures_tested=False),[
variant('text_document_summary_ja','ja',r'''
以下は架空の地域博物館の試行報告です。事実、解釈、次の提案を分けて短くまとめてください。
博物館は土曜日の資料解説会を6回開き、各回の定員を12人にしました。重複を除く90人が予約を希望し、72人が予約を取得、18人は満席のため予約できませんでした。予約者のうち64人が参加し、8人は連絡なく欠席しました。博物館は待機希望者への連絡許可を取っていなかったため、当日の空席を待機者へ案内しませんでした。
参加者からの任意アンケートには44人が回答し、38人が「良かった」、6人が「普通」と答えました。未回答者の評価は記録されていません。職員には各回30分の追加準備が発生しました。担当者は、予約確認の通知と、連絡を希望する人だけを登録する待機リストを次回の案として挙げています。どちらの対策も、まだ試していません。
参加状況の割合を述べる場合は分母を明記し、アンケートの結果を参加者全員や地域全体へ一般化しないでください。欠席の理由はこの報告に書かれていないことにも注意してください。
''',r'''
確認された結果として、6回で計72席が用意され、72人が予約し、64人が参加しました。予約者を分母とする参加率は64/72で約88.9%、無連絡欠席の割合は8/72で約11.1%です。予約を希望した90人のうち18人は、満席のため予約を取得できませんでした。職員の追加準備は30分×6回で計180分です。
アンケートでは回答者44人中38人が「良かった」、6人が「普通」と答えました。この結果は回答者についての記録であり、未回答者20人の評価は分かりません。参加者全員や地域全体の満足度を示したとは言えません。
運用上は、待機者への連絡許可を集めていなかったため、当日の空席を案内できませんでした。ただし、8人が欠席した理由は記録されておらず、通知不足が原因だと断定できません。予約確認の通知と、同意を得た待機リストは次回の提案です。まだ試されていないため、欠席や空席をどの程度減らすかは今後の確認事項です。
'''),
variant('text_document_summary_en','en',r'''
The following is a fictional community museum's pilot report. Summarize the observed facts, interpretations, and proposed next steps separately.
The museum held six Saturday collection talks, each with twelve seats. Ninety distinct people requested reservations. Seventy-two obtained reservations and eighteen could not because all seats were booked. Of those with reservations, sixty-four attended and eight did not attend or contact the museum. The museum had not collected permission to contact people waiting for a seat, so it did not offer the empty seats to them on the day.
Forty-four attendees answered an optional survey: thirty-eight selected "good" and six selected "neutral." The opinions of nonrespondents were not recorded. Staff spent an additional thirty minutes preparing each session. The organizer proposed reservation reminders and a waiting list limited to people who agree to be contacted. Neither measure has been tried yet.
State denominators whenever reporting attendance percentages. Do not generalize the survey to every attendee or to the whole community. Also note that the report does not state why the eight people were absent.
''',r'''
The observed results are six sessions with seventy-two seats in total, seventy-two reservations, and sixty-four attendees. Attendance among reservation holders was 64/72, approximately 88.9%, while unannounced absence was 8/72, approximately 11.1%. Eighteen of the ninety people requesting reservations could not obtain one because the sessions were fully booked. Additional staff preparation totaled thirty minutes times six, or 180 minutes.
Among forty-four survey respondents, thirty-eight selected "good" and six selected "neutral." Those are results for respondents; the views of the other twenty attendees are unknown. The survey does not establish satisfaction among all attendees or across the community.
Operationally, the museum could not offer the empty seats to waiting people because it had not collected permission to contact them. However, the reasons for the eight absences were not recorded, so a lack of reminders cannot be established as their cause. Reservation reminders and an opt-in waiting list are proposals for the next pilot. Since neither has been tried, their effects on absence and unused seats remain to be measured.
''')])

family('mixed_build_log','mixed_workflow','train',
       dict(language_standard='C++17',missing_translation_unit='parser.cpp',
            declared_and_defined_signature='int parse_row(const std::string&)',
            unrelated_warning='unused variable scratch',correct_sources=['main.cpp','store.cpp','parser.cpp']),[
variant('mixed_build_log_ja_en_log','ja+en+log',r'''
小さなC++のテストプログラムがlinkで止まりました。以下は説明用に作ったファイル構成と、型名を一部短縮したログです。原因を、見えている証拠だけから日本語で説明してください。安全に検証できる最小の修正案も示してください。
parse.h は int parse_row(const std::string& line); を宣言します。main.cpp はparse.hをincludeしてparse_rowを呼びます。parser.cppは同じ宣言の関数本体を定義しており、store.cppには別の保存処理だけがあります。全ファイルは同じC++17設定でcompileする前提です。
実行したcommand:
c++ -std=c++17 main.cpp store.cpp -o catalog_test
ログ:
main.cpp: warning: unused variable 'scratch'
linker: undefined reference to 'parse_row(std::string const&)'
linker: error: link failed
関数名やnamespaceの違いを示す追加情報はありません。warningを消すだけでlinkが直ると決めつけず、宣言と定義とlink入力の関係を説明してください。パッケージ更新や大規模なbuild設定の書き換えは、現時点の証拠に必要な範囲へ限定してください。
''',r'''
提示されたcommandには、parse_rowの定義があるparser.cppが入っていません。main.cppから宣言は見えているため呼出しをcompileできても、link時に必要な定義を探せず、undefined referenceになる説明と一致します。store.cppにはその定義がないという前提なので、これを加えても不足は埋まりません。
最小の確認は、同じ設定でparser.cppもlink入力に加えることです。
c++ -std=c++17 main.cpp store.cpp parser.cpp -o catalog_test
別々にcompileする方式なら、parser.cppから作ったobjectも最終linkへ渡します。修正後はlinkが成功することと、parse_rowを使う既存の入力例で出力が期待どおりになることを確認します。
unused variableのwarningは使われていない変数に関する別の指摘です。その整理は可能ですが、parser.cppの定義がlinkへ入っていない問題を直接解決しません。今回の情報からABIやnamespaceの不一致を確定することもできません。まず明示されている入力不足を一つ直し、残ったエラーがあれば、その新しいログで次を判断します。
''')])

family('mixed_config_change','mixed_workflow','train',
       dict(timeout_ms=1500,retry_count=3,backoff_ms=[100,200,400],
            maximum_attempts=4,all_timeout_budget_ms=6700,
            budget_excludes=['request overhead','scheduler delay']),[
variant('mixed_config_change_ja_json','ja+json',r'''
架空のHTTP clientの設定レビューをお願いします。このclientのschemaでは、timeout_msは100〜30000の整数、retry_countは0〜5の整数、backoff_msは非負整数の配列です。backoff_msの要素数はretry_countと一致させます。endpointは文字列です。初回要求に加えてretry_count回まで再試行し、各再試行の直前に配列の対応する待機時間を使います。
現在のJSON:
{"endpoint":"https://example.invalid/export","timeout_ms":"1.5","retry_count":3,"backoff_ms":[100,200]}
意図は、1回のtimeoutを1.5秒、再試行を3回、待機を順に100、200、400ミリ秒にすることです。endpointは変更しません。意図に合う完全なJSONを示し、型、単位、配列の長さの三つを確認してください。すべての試行がtimeoutまでかかると仮定し、要求処理の付加費用やschedulerの遅れを除いた合計時間も計算してください。その仮定付きの値を、実測した応答時間として扱わないでください。
''',r'''
意図に合う設定は次のとおりです。
{"endpoint":"https://example.invalid/export","timeout_ms":1500,"retry_count":3,"backoff_ms":[100,200,400]}
timeout_msは文字列の"1.5"から整数の1500へ変えます。単位はミリ秒なので、1.5秒をそのまま1.5という数で入れるのでもありません。1500は指定された100〜30000の範囲内です。retry_countは整数3を維持し、backoff_msに400を加えることで、再試行3回と要素数3を一致させます。待機値もすべて非負整数です。
最大試行数は初回1回と再試行3回を合わせた4回です。すべてが1500 msのtimeoutまでかかるという仮定では、試行部分が4×1500=6000 ms、待機部分が100+200+400=700 ms、合計6700 msとなります。これは指定した単純な計算モデルの値です。要求の準備、その他の処理やschedulerの遅れを含む実測時間を保証するものではありません。まずschemaの検査を通し、実際の要求では試行回数と待機の記録を確認します。
''')])

family('mixed_table_reconciliation','mixed_workflow','validation',
       dict(shipped_by_sku=dict(A=17,B=7),accepted_unique_returns=dict(A=2,B=1),
            duplicate_return_id='r01',pending_return_id='r03',
            expected_net_by_sku=dict(A=15,B=6),expected_total=21,
            reported_net_by_sku=dict(A=13,B=6),reported_total=19),[
variant('mixed_table_reconciliation_ja_en-columns','ja+en-columns',r'''
次の架空の出荷・返品表を突き合わせてください。数量はすべて整数の個数です。出荷はshipment_idごとに一意です。返品はreturn_idを一意な識別子として扱い、同じIDかつ同じ内容の再送は一度だけ数えます。同じIDで内容が矛盾する場合は黙って片方を選ばず、確認対象とします。statusがacceptedの返品だけを差し引き、pendingはまだ差し引きません。
shipments:
shipment_id,sku,quantity
s01,A,12
s02,B,7
s03,A,5
returns:
return_id,shipment_id,sku,quantity,status
r01,s01,A,2,accepted
r02,s02,B,1,accepted
r01,s01,A,2,accepted
r03,s03,A,1,pending
現在の報告書はA=13、B=6、合計19としています。正しい品目別の正味数量と合計を求め、報告書との差がどの行の扱いで説明できるかを示してください。pendingを確定返品として扱わず、返品行の単純な合計だけで判定しないでください。
''',r'''
出荷数量はAが12+5=17個、Bが7個です。返品ではr01が同じ内容で二度現れますが、規則上は一件として数えるため、Aから差し引く確定返品は2個です。r02はBの確定返品1個です。r03はpendingなので、この時点ではAから差し引きません。
正味数量はA=17-2=15個、B=7-1=6個で、合計21個です。現在の報告書のA=13は、r01の2個を二度差し引いた17-2-2=13という計算と一致します。Bは一致しており、合計も正しい21に対して2個少なくなっています。
今回のr01はIDだけでなく内容も一致するので、同一返品の再送として一度にまとめられます。IDが同じでも数量や品目が異なる行があれば、この処理で自動的に片方を捨ててはいけません。元の行と除外理由を残し、出荷IDとの対応、重複判定、acceptedだけの集計を別々に確認すると再現可能になります。pendingが後でacceptedへ変わる場合は、その時点の状態に基づいて改めて集計します。
''')])

family('mixed_api_incident','mixed_workflow','holdout',
       dict(affected_request='q42',affected_node='N2',affected_endpoint='POST /export',
            recorded_error_stage='parse_config',recorded_missing_key='archive_path',
            comparison_request='q44',same_payload_hash='p9',
            output_open_attempted_in_q42=False,
            global_endpoint_outage_proven=False),[
variant('mixed_api_incident_ja_en_http','ja+en+http',r'''
テスト環境でexport要求が一度だけ500になりました。以下は架空のログで、同じrequest_idの行は同じ要求です。N1とN2は同じアプリの二つのnodeです。payload_hashは要求本文の識別用で、秘密情報ではありません。記録のない内部動作は推測で確定しないでください。
09:14:01 node=N1 request=q41 method=GET path=/items status=200 total_ms=19
09:14:03 node=N2 request=q42 method=POST path=/export payload_hash=p9
09:14:03 node=N2 request=q42 stage=parse_config result=error key=archive_path reason=missing
09:14:03 node=N2 request=q42 stage=open_output skipped=true
09:14:03 node=N2 request=q42 status=500 total_ms=3
09:14:05 node=N2 request=q43 method=GET path=/items status=200 total_ms=21
09:14:07 node=N1 request=q44 method=POST path=/export payload_hash=p9 status=202 queued=true total_ms=11
最も強く支持される説明、まだ分からないこと、次に行う最小の比較を日本語で整理してください。全APIが停止した、disk権限が原因だ、などと記録以上に断定しないでください。修正を確認するときには、単にHTTP statusだけを見る場合の限界にも触れてください。
''',r'''
最も直接の根拠は、N2のq42で設定読込の段階にarchive_pathの欠落が記録されていることです。その後のopen_outputはskipされているため、この要求が出力先を開こうとして権限エラーになった、という説明はログに支えられていません。まずN2でexportに使う設定が、期待どおり読み込まれているかを調べるのが適切です。
N2の別要求q43は/itemsで200を返し、N1のq44は同じpayload_hashの/exportを202で受け付けています。このため全API停止とは言えません。ただし、N1とN2の全環境差や、設定が欠けた経緯までは分かりません。202とqueued=trueも、export処理そのものの完了を示してはいません。
次は同じ版の設定と実際に読み込まれたarchive_pathをnode間で照合し、同じ本文のexportを各nodeで一度ずつ比較します。修正後は設定読込を通ること、出力先を開く段階へ進むこと、処理が完了して期待する成果物が得られることまで確認します。成功statusだけで下流の完了を省略せず、元の失敗requestと再試験を別の記録として残します。
''')])

# Code variants are added below. They are original reference programs, not engine code.

group_py = r'''
def ordered_groups(rows: list[tuple[str, int]]) -> list[tuple[str, list[int]]]:
    """Group values without sorting keys, removing duplicates, or changing rows.

    Keys, including the empty string, are compared exactly. Values are retained
    in input order. Valid inputs contain string keys and signed 32-bit integers.
    """
    positions: dict[str, int] = {}
    result: list[tuple[str, list[int]]] = []
    for key, value in rows:
        if key not in positions:
            positions[key] = len(result)
            result.append((key, []))
        result[positions[key]][1].append(value)
    return result
'''
group_cpp = r'''
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using Row = std::pair<std::string, std::int32_t>;
using Group = std::pair<std::string, std::vector<std::int32_t>>;

std::vector<Group> ordered_groups(const std::vector<Row>& rows) {
    // Group order is first appearance; value order and duplicates are retained.
    // Empty keys are valid. No normalization or sorting is performed.
    std::unordered_map<std::string, std::size_t> positions;
    std::vector<Group> result;
    for (const auto& row : rows) {
        auto found = positions.find(row.first);
        if (found == positions.end()) {
            const std::size_t index = result.size();
            positions.emplace(row.first, index);
            result.emplace_back(row.first, std::vector<std::int32_t>{});
            found = positions.find(row.first);
        }
        result[found->second].second.push_back(row.second);
    }
    return result;
}
'''
group_prompt = r'''
Write a {language} implementation of ordered_groups for a list of (key, value) rows. Each key is a string, including possibly the empty string; each value is a signed 32-bit integer. Inputs already have valid types. Return a list of groups. Each group contains a key and the list of all values seen for that key. The groups must appear in the order in which their keys first appeared. Values within a group keep input order, including repeated and negative values. Compare keys exactly: do not case-fold, normalize Unicode, or sort. Do not modify the input. Empty input produces an empty list.
For [("north", 4), ("", 9), ("north", -2), ("east", 5), ("north", 4)], the result must be [("north", [4,-2,4]), ("", [9]), ("east", [5])]. Use only the standard library and explain how the implementation preserves both kinds of order. In particular, do not replace a group's value list with a sum or a set. Avoid scanning all earlier groups for every input row.
'''
group_intro = 'Keep a map from each key to its position in an output list. Append a group only at the first appearance of its key, then append every value to that group. This preserves duplicate values and both required orders without changing the input.'
family('code_stable_grouping','code_spec','train',
       dict(function='ordered_groups',input_domain='valid string keys and signed 32-bit integer values',
            empty_key_valid=True,unicode_normalization=False,deduplicate_values=False,
            preserve_input=True,group_order='first key appearance',value_order='input order',
            example_rows=[['north',4],['',9],['north',-2],['east',5],['north',4]],
            example_output=[['north',[4,-2,4]],['',[9]],['east',[5]]]),[
variant('code_stable_grouping_python','python',group_prompt.format(language='Python 3'),
        '```python\n'+group_py.strip()+'\n```\n\n'+group_intro,code=group_py,language='python'),
variant('code_stable_grouping_cpp','cpp',group_prompt.format(language='C++17'),
        '```cpp\n'+group_cpp.strip()+'\n```\n\n'+group_intro,code=group_cpp,language='cpp')])

interval_py = r'''
def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return the union of half-open intervals as sorted, separated intervals.

    Reject reversed endpoints, ignore empty intervals, and merge touching ones.
    Endpoints are signed 64-bit integers; no arithmetic on them is necessary.
    """
    work: list[tuple[int, int]] = []
    for start, end in intervals:
        if start > end:
            raise ValueError("reversed interval")
        if start < end:
            work.append((start, end))
    work.sort()
    merged: list[tuple[int, int]] = []
    for start, end in work:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
    return merged
'''
interval_cpp = r'''
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

using Interval = std::pair<std::int64_t, std::int64_t>;

std::vector<Interval> merge_intervals(const std::vector<Interval>& intervals) {
    // Validate before sorting a private copy. The caller's input is unchanged.
    std::vector<Interval> work;
    for (const auto& interval : intervals) {
        if (interval.first > interval.second) {
            throw std::invalid_argument("reversed interval");
        }
        if (interval.first < interval.second) {
            work.push_back(interval);
        }
    }
    std::sort(work.begin(), work.end());
    std::vector<Interval> merged;
    for (const auto& interval : work) {
        if (merged.empty() || interval.first > merged.back().second) {
            merged.push_back(interval);
        } else {
            merged.back().second = std::max(merged.back().second, interval.second);
        }
    }
    return merged;
}
'''
interval_prompt = r'''
Implement merge_intervals in {language}, using only the standard library. An input item (a,b) represents the half-open interval [a,b), and endpoints are signed 64-bit integers. Reject any item with a>b by raising the language's ordinary invalid-input exception. Ignore empty intervals where a=b. Return the union as a list sorted by start, merging both overlapping and touching intervals. The returned intervals must be nonempty and must have a strict gap between neighbors. Do not mutate the input and do not add or subtract from endpoints, so extreme valid endpoints do not require overflow-prone arithmetic.
For [(5,7),(1,3),(3,5),(10,10)], the result is [(1,7)]. An empty input or an input containing only empty intervals gives an empty result. Contained intervals and duplicates must not create extra output entries. Include the implementation and explain why a sorted sweep handles these cases, especially the distinction between start>end and start=end.
'''
interval_intro = 'First reject reversed intervals and copy only nonempty ones. Sort that private list, then sweep it once. A new interval starts a separate output entry only when its start is strictly beyond the previous end; equality means the intervals touch and should be merged.'
family('code_interval_merge','code_spec','train',
       dict(function='merge_intervals',endpoint_domain='signed 64-bit integers',half_open=True,
            reversed='exception',empty='ignore',touching='merge',preserve_input=True,
            example_input=[[5,7],[1,3],[3,5],[10,10]],example_output=[[1,7]]),[
variant('code_interval_merge_python','python',interval_prompt.format(language='Python 3'),
        '```python\n'+interval_py.strip()+'\n```\n\n'+interval_intro,code=interval_py,language='python'),
variant('code_interval_merge_cpp','cpp',interval_prompt.format(language='C++17'),
        '```cpp\n'+interval_cpp.strip()+'\n```\n\n'+interval_intro,code=interval_cpp,language='cpp')])

queue_py = r'''
from typing import Optional


class RingQueue:
    """A fixed-capacity FIFO of integers; failures leave the queue unchanged."""

    def __init__(self, capacity: int):
        if not 1 <= capacity <= 1024:
            raise ValueError("capacity must be in 1..1024")
        self._data = [0] * capacity
        self._head = 0
        self._count = 0

    def size(self) -> int:
        return self._count

    def try_push(self, value: int) -> bool:
        if self._count == len(self._data):
            return False
        tail = (self._head + self._count) % len(self._data)
        self._data[tail] = value
        self._count += 1
        return True

    def try_pop(self) -> Optional[int]:
        if self._count == 0:
            return None
        value = self._data[self._head]
        self._head = (self._head + 1) % len(self._data)
        self._count -= 1
        return value
'''
queue_cpp = r'''
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

class RingQueue {
public:
    explicit RingQueue(std::int64_t capacity) {
        if (capacity < 1 || capacity > 1024) {
            throw std::invalid_argument("capacity must be in 1..1024");
        }
        data_.resize(static_cast<std::size_t>(capacity));
    }

    std::size_t size() const { return count_; }

    bool try_push(std::int64_t value) {
        if (count_ == data_.size()) {
            return false;
        }
        const std::size_t tail = (head_ + count_) % data_.size();
        data_[tail] = value;
        ++count_;
        return true;
    }

    std::optional<std::int64_t> try_pop() {
        if (count_ == 0) {
            return std::nullopt;
        }
        const std::int64_t value = data_[head_];
        head_ = (head_ + 1) % data_.size();
        --count_;
        return value;
    }

private:
    std::vector<std::int64_t> data_;
    std::size_t head_ = 0;
    std::size_t count_ = 0;
};
'''
queue_prompt = r'''
Write a {language} RingQueue class for signed 64-bit integer values. Inputs already have valid integer types, with booleans excluded. The capacity argument and pushed values are representable as signed 64-bit integers; checking those type and representation preconditions is not required. Capacity is fixed at construction and must be from 1 through 1024; reject capacities outside that range. Provide size(), try_push(value), and try_pop(). A push to a full queue returns false and must not change the queue. A pop from an empty queue returns an explicit empty optional result, not the integer zero. All successful operations follow FIFO order. Zero, negative values, and repeated values are legitimate stored data.
Use a ring buffer with an index for the oldest element and a count, rather than shifting stored elements after each pop. For capacity two, pushing 4 and 9 succeeds, pushing 12 fails, popping returns 4, pushing -3 succeeds, and the next two pops return 9 and -3. Explain how wraparound works and why a failed operation leaves the state intact. Use only the standard library and keep the interface independent of printing or file I/O.
'''
queue_intro = 'Store the head index and the number of elements separately. The next insertion position is (head + count) modulo capacity. Check full or empty conditions before changing any state, and use an optional result to distinguish an empty queue from a stored zero.'
family('code_bounded_queue','code_spec','validation',
       dict(class_name='RingQueue',capacity_domain=[1,1024],value_domain='signed 64-bit integers',
            input_type_precondition='valid integer arguments, excluding bool; capacity and pushed values representable as signed 64-bit',
            runtime_validation_required='capacity range only; type and value-range preconditions already hold',
            full_push='false and unchanged',empty_pop='empty optional and unchanged',
            order='FIFO',zero_is_data=True,storage='ring buffer'),[
variant('code_bounded_queue_python','python',queue_prompt.format(language='Python 3'),
        '```python\n'+queue_py.strip()+'\n```\n\n'+queue_intro,code=queue_py,language='python'),
variant('code_bounded_queue_cpp','cpp',queue_prompt.format(language='C++17'),
        '```cpp\n'+queue_cpp.strip()+'\n```\n\n'+queue_intro,code=queue_cpp,language='cpp')])

lines_py = r'''
from typing import Optional


class LineBuffer:
    """Split a byte stream at LF without decoding or changing any other byte."""

    def __init__(self):
        self._pending = bytearray()

    def feed(self, chunk: bytes) -> list[bytes]:
        lines: list[bytes] = []
        for value in chunk:
            if value == 10:
                lines.append(bytes(self._pending))
                self._pending.clear()
            else:
                self._pending.append(value)
        return lines

    def finish(self) -> Optional[bytes]:
        if not self._pending:
            return None
        line = bytes(self._pending)
        self._pending.clear()
        return line
'''
lines_cpp = r'''
#include <optional>
#include <string>
#include <string_view>
#include <vector>

class LineBuffer {
public:
    // Strings hold bytes. UTF-8 decoding is intentionally outside this class.
    std::vector<std::string> feed(std::string_view chunk) {
        std::vector<std::string> lines;
        for (char value : chunk) {
            if (value == '\n') {
                lines.push_back(pending_);
                pending_.clear();
            } else {
                pending_.push_back(value);
            }
        }
        return lines;
    }

    std::optional<std::string> finish() {
        if (pending_.empty()) {
            return std::nullopt;
        }
        std::string line = pending_;
        pending_.clear();
        return line;
    }

private:
    std::string pending_;
};
'''
lines_prompt = r'''
Implement a {language} LineBuffer that receives arbitrary byte chunks from a UTF-8 text stream. A chunk may end in the middle of a multibyte character. feed(chunk) returns every complete line terminated by byte LF (0x0A), without the LF itself. Preserve all other bytes exactly, including CR (0x0D); this task does not normalize CRLF. Empty lines between LF bytes must be returned. Keep the unfinished final line across calls. finish() returns that final line once if it is nonempty, then clears it; otherwise it returns an empty optional result. A stream ending in LF does not acquire an additional empty line at finish(). Empty input chunks have no effect.
The full input is assumed to be valid UTF-8, but this class must not decode individual chunks or validate UTF-8. Return byte strings, allowing a caller to decode each completed line later. Distinguish an emitted empty line from no line. Use only the standard library, process each input byte once, and explain why splitting a multibyte character across two chunks does not lose or duplicate its bytes.
'''
lines_intro = 'Buffer bytes until an LF is encountered, then emit the accumulated bytes and clear the buffer. Do not decode incoming chunks. A split UTF-8 character stays in the buffer byte for byte; the LF separator is handled independently. Preserve CR and emit empty LF-terminated lines as real empty byte strings.'
family('code_utf8_line_reader','code_spec','holdout',
       dict(class_name='LineBuffer',input='byte chunks; full stream assumed valid UTF-8',
            separator_byte=10,preserve_CR=True,decode=False,validate_UTF8=False,
            empty_lines='emit',finish='nonempty remainder once; empty optional otherwise',
            empty_chunk='no effect',complexity='each input byte visited once'),[
variant('code_utf8_line_reader_python','python',lines_prompt.format(language='Python 3'),
        '```python\n'+lines_py.strip()+'\n```\n\n'+lines_intro,code=lines_py,language='python'),
variant('code_utf8_line_reader_cpp','cpp',lines_prompt.format(language='C++17'),
        '```cpp\n'+lines_cpp.strip()+'\n```\n\n'+lines_intro,code=lines_cpp,language='cpp')])


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def main():
    from collections import Counter
    ids=[]
    for fam in families:
        for v in fam['variants']:
            ids.append(v['case_id'])
            for field in ('prompt_utf8','reference_utf8'):
                text=v[field]
                stem=field.removesuffix('_utf8')
                v[stem+'_sha256']=digest(text)
                v[stem+'_utf8_bytes']=len(text.encode())
                v[stem+'_codepoints']=len(text)
            if 'reference_code_utf8' in v:
                v['reference_code_sha256']=digest(v['reference_code_utf8'])
    counts=Counter(f['split'] for f in families for v in f['variants'])
    assert len(families)==12 and len(ids)==len(set(ids))==20
    assert dict(counts)=={'train':10,'validation':5,'holdout':5}
    corpus=dict(schema=1,date_jst='2026-11-17',
        status='original_text_and_reference_ready; untokenized; no model execution',
        purpose='Small static-bank selection pilot after capacity feasibility; train selects placement, not model weights',
        origin='Original synthetic tasks and reference solutions authored for this review',
        pin_for_review='fa6671f7b5281bad28947fe5324e2495e792a86f',
        calibration_cases_reused=False,calibration_routes_used_for_selection=False,
        expected_case_count=20,expected_family_count=12,cases_by_split=dict(counts),
        tokenizer_preflight_required=True,model_token_counts=None,
        prefix_policy='Full rendered user prompt with the actual chosen chat template; no truncation to 64 prompt tokens',
        continuation_policy='First 64 fixed-reference input tokens after a verified assistant boundary; not whole-answer timing',
        rendering_note='Use the production-equivalent chat template if studying chat. A raw-text fallback is a separately named experiment. Store the exact rendering, IDs, boundary and hashes; do not assume tokenizing pieces and concatenating IDs equals tokenizing their concatenated text.',
        too_short_policy='Fail token preflight and naturally extend a reference before corpus freeze; do not repeat/pad text to reach 64 tokens',
        statistical_unit='family; translations/ports/windows/turns stay together',
        weights=dict(primary='report ja/en/mixed/python/cpp separately',diagnostic_macro='equal lane',production_weights=None),
        model_quality_evaluated=False,actual_gpu_bank_admission=False,
        families=families)
    (OUT/'PILOT_CASES_JA_EN_CODE.json').write_text(json.dumps(corpus,ensure_ascii=False,indent=2)+'\n')
    stages=[
       dict(stage='capacity',cases=[],purpose='Establish fresh-free/guard feasibility for a bounded native bank before collecting the pilot; no routing choice or completed hybrid path is required at this stage'),
       dict(stage='train_first5',family_ids=['text_cache_budget','mixed_build_log','code_stable_grouping'],
            purpose='Small initial training route screen; form a transparent candidate and probe costs or a bounded positive path if available. Unmeasured utility stays unknown; may stop if feasibility or measured benefit is absent'),
       dict(stage='train_remaining5',family_ids=['text_unit_conversion','mixed_config_change','code_interval_merge'],
            purpose='Complete the training split only if the first stage justifies proceeding'),
       dict(stage='validation5',family_ids=['text_plan_constraints','mixed_table_reconciliation','code_bounded_queue'],
            purpose='Compare a predeclared small candidate set; then freeze bank, cost model, budget, weights and exclusions'),
       dict(stage='holdout5',family_ids=['text_document_summary','mixed_api_incident','code_utf8_line_reader'],
            purpose='One frozen candidate evaluation; baseline and actual hybrid routes/output/timing on identical input IDs')]
    for stage in stages:
        if 'family_ids' in stage:
            stage['cases']=[v['case_id'] for f in families if f['family_id'] in stage['family_ids'] for v in f['variants']]
            assert len(stage['cases'])==5
        stage['calls_if_64_per_case']=64*len(stage['cases'])
        stage['records_if_observed_complete']=37*stage['calls_if_64_per_case']
    manifest=dict(schema=1,corpus_file='PILOT_CASES_JA_EN_CODE.json',
        corpus_sha256=hashlib.sha256((OUT/'PILOT_CASES_JA_EN_CODE.json').read_bytes()).hexdigest(),
        stages=stages,full_pilot_calls=1280,full_pilot_records=47360,records_per_session=2368,
        cases_by_split=dict(counts),all_text_frozen_before_routes=True,token_manifest_still_required=True,
        one_session_per_case=True,original_six_calibration_cases_excluded=True,
        additional_windows_remain_in_same_family=True,
        freeze_before_holdout=['native_bank_layer_expert_ids','actual_triplet_bytes','cost_model_version','budget','candidate_objective','weights','failure_exclusions','rendered_text_and_token_manifest_hashes'],
        current_caller_is_not_drop_in_compatible='Current calibration caller hardcodes prefix64 and a raw text field; full prompt/reference boundary and variable expected positions require a bounded extension.',
        unknown_fields=['model_token_counts','rendered_prefix_token_counts','GPU_ready_cost','CPU_remaining_cost','saved_time','positive_bank_admission'])
    (OUT/'PILOT_STAGE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'families':len(families),'cases':len(ids),'splits':dict(counts),'stages':[{'stage':s['stage'],'cases':len(s['cases'])} for s in stages]},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

