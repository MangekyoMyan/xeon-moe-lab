GPU workspace / native-bank capacity / train_first5 — 2026-11-17
=============================================================
RESULTS_JA.txt: 事実、前回判断の訂正、数値と未実施事項。
CONSULT_PRO_JA.txt: 次の小さいCPU/GPU分担試験の具体的相談。

root driver/build/run_capacity: bankなしfreshU1024のM0..M3とshort64/long1024。
bank-capacity/: 旧hot-backend create/finishだけの容量fixture。モデルへは未組込み。
pilot-capture/: bankなしfull chat prompt + 64 reference tokens、first5 trainだけ。
PILOT_TOKENS.json: 元20caseのAPI template/tokenizer前処理のみ。全splitと境界を保持。
TRAIN5_PROPOSAL.json / TRAIN5_CANDIDATE.plan: fitted training proposal、未確保/未適用。
BANK_*AUDIT_TERRA.txt / PILOT_ACTUAL_AUDIT_TERRA.txt: 独立raw監査。
RESTORED_RUNTIME.json: 最後の通常API復帰。RESTORED_BEFORE_PILOT.jsonは途中の復帰履歴。

全rawの内容は省略せず共有。公開時だけhealth.jsonをguard-health.jsonへ改名し、
runtime server-config.jsonは除外。安全なcommand/envはLAUNCH_MANIFESTに入る。
RAW_MAPPINGとFILES_SHA256で原本との一致を検査可能。ELF/model weights/credentialsは含めない。
checkpointsは実行前のbuild/metadata修正履歴であり、現在使うbinaryは各top-level BUILDが指定する。

保存物だけの再検証はreplay_shared.pyを参照。model/GPU/networkを使わず一時copyで検査する。
これは通常の閲覧者向けREADMEではなく、実験/LLMレビュー用の補足資料。
今ユーザーに実行をお願いしているcommandはありません。通常APIは元の構成に復帰済みです。
