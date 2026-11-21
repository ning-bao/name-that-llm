"""Author the independent prompt bank and collection metadata specification.
No model queries, training, or competition submissions are performed.
"""
import json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'deliverables';OUT.mkdir(exist_ok=True)

# Independent prompts written for this report. No competition passage is copied.
# Each prompt is a separate request in a fresh conversation, across all accessible
# exact nominal target models. H prompts are reserved for evaluation.
prompts=[
('N01','train','physics','conceptual explanation','40-70','Explain why a metal handrail and a wooden handrail can feel different in temperature even when they have been in the same room overnight. Write one self-contained paragraph of about 40-70 words.'),
('N02','train','space','evocative exposition','90-130','Describe how the shape of a distant planet might be inferred from a tiny change in starlight. Make the explanation accessible and convey the difficulty of learning about something that cannot be seen directly. Write one paragraph of about 90-130 words.'),
('N03','train','life science','conceptual explanation','160-210','Why does a healing skin wound sometimes leave a scar instead of restoring exactly what was there before? Explain the biological tradeoff to an interested general reader in one paragraph of about 160-210 words.'),
('N04','train','mathematics','reflective argument','220-250','Two maps can represent the same city while making it feel like two different places. Use this idea to explain what mathematical abstraction preserves and what it leaves out. Write one self-contained paragraph of about 220-250 words.'),
('N05','train','computing','conceptual explanation','40-70','A shared document can show conflicting edits when two people work offline. Explain why resolving the conflict involves a choice rather than simply discovering which computer is correct. Write one paragraph of about 40-70 words.'),
('N06','train','philosophy','reflective argument','90-130','Can an apology still matter if the person receiving it never forgives? Explore one tension in this question without turning the answer into a list. Write one paragraph of about 90-130 words.'),
('N07','train','history/culture','evocative exposition','160-210','Describe how an old harbor can preserve the history of trade even after its warehouses have become homes and cafes. Connect physical details with the movement of people and ideas. Write one paragraph of about 160-210 words.'),
('N08','train','creative prose','creative scene','220-250','Write a scene in which a night-shift conservator notices a fresh fingerprint inside a sealed museum display. Keep the explanation unresolved and let the setting carry the tension. Use one paragraph of about 220-250 words.'),
('N09','train','life science','evocative exposition','40-70','Describe a seed waiting through a dry season, connecting its apparent stillness with the living processes that make future growth possible. Write one paragraph of about 40-70 words.'),
('N10','train','mathematics','conceptual explanation','90-130','A fair coin has landed heads six times in a row. Explain why the next toss and our suspicion about the coin are two different questions. Write one paragraph of about 90-130 words.'),
('N11','train','space','conceptual explanation','160-210','How could astronomers learn that a small moon contains an ocean without drilling through its surface? Discuss the kinds of indirect evidence and what would remain uncertain. Write one paragraph of about 160-210 words.'),
('N12','train','mind/behavior','reflective argument','220-250','Why can a familiar place feel different when visited after a long absence, even if very little in it has changed? Explore the relationship between expectation, attention, and memory. Write one paragraph of about 220-250 words.'),
('N13','train','history/culture','conceptual explanation','40-70','Why might two careful historians disagree about the significance of the same surviving letter? Write one self-contained paragraph of about 40-70 words.'),
('N14','train','earth/environment','evocative exposition','90-130','Describe a river delta as a place that is always being built and erased. Explain how sediment, water, and human decisions shape this changing boundary. Write one paragraph of about 90-130 words.'),
('N15','train','computing','reflective argument','160-210','A computer program gives the right answer on every example its developer tried. Explain why that success is different from knowing the program is correct, using an accessible analogy. Write one paragraph of about 160-210 words.'),
('N16','train','philosophy','reflective argument','220-250','A community chooses to preserve an uncomfortable monument while changing the explanation beside it. Explore what this choice suggests about remembering, honoring, and judging the past. Write one paragraph of about 220-250 words.'),
('N17','train','creative prose','creative scene','40-70','Write a quiet scene in which a person opens a repaired umbrella on a perfectly clear day. Let a small physical detail suggest why the act matters. Use one paragraph of about 40-70 words.'),
('N18','train','physics','conceptual explanation','90-130','Why can a smooth-looking sheet of ice break suddenly after supporting a weight for some time? Explain how small imperfections can matter without presenting a mathematical derivation. Write one paragraph of about 90-130 words.'),
('N19','train','society/economics','reflective argument','160-210','A town removes the fare from its buses, but some residents still cannot use them. Explain what this reveals about the difference between price and access. Write one paragraph of about 160-210 words.'),
('N20','train','arts/language','evocative exposition','220-250','Describe what a translator must decide when a short joke depends on a word with two meanings. Explore what might be preserved even when the original wording cannot survive. Write one paragraph of about 220-250 words.'),
('N21','train','space','evocative exposition','40-70','Describe the silence inside a spacecraft while its instruments record a violent storm on a planet below. Keep the physical setting plausible. Write one paragraph of about 40-70 words.'),
('N22','train','life science','conceptual explanation','90-130','How can a species become common in a new habitat without being unusually strong or intelligent? Explain the role of relationships and opportunities in one paragraph of about 90-130 words.'),
('N23','train','mathematics','conceptual explanation','160-210','A voting rule seems fair in several simple examples but behaves strangely when a third candidate enters. Explain why a few appealing examples cannot settle whether a rule is fair. Write one paragraph of about 160-210 words.'),
('N24','train','creative prose','creative scene','220-250','Write a scene in which a person responsible for winding a public clock discovers that the clock has continued working during a week-long absence. Build the scene around observation rather than a final explanation. Use one paragraph of about 220-250 words.'),
('H01','external_holdout','physics','conceptual explanation','40-70','Why does a drop of water cling to the underside of a leaf for a while before falling? Explain the competing effects in one paragraph of about 40-70 words.'),
('H02','external_holdout','computing','conceptual explanation','90-130','A search engine returns a highly relevant document that contains an incorrect claim. Explain why finding useful information and deciding whether it is true are separate problems. Write one paragraph of about 90-130 words.'),
('H03','external_holdout','history/culture','evocative exposition','160-210','Describe how a traditional weaving pattern can carry practical knowledge about a landscape and its seasons. Explain what an outsider might miss by seeing only decoration. Write one self-contained paragraph of about 160-210 words.'),
('H04','external_holdout','philosophy','reflective argument','220-250','Should someone receive more moral credit for resisting a strong temptation than for doing the same right thing without hesitation? Examine what effort can and cannot tell us about character. Write one paragraph of about 220-250 words.'),
('H05','external_holdout','life science','conceptual explanation','90-130','How can a living organism maintain a stable internal temperature while its surroundings keep changing? Explain the idea of regulation to a general reader in one paragraph of about 90-130 words.'),
('H06','external_holdout','creative prose','creative scene','160-210','Write a warm, understated scene in which a child teaches a grandparent to identify neighborhood birds by listening. Let a small misunderstanding reveal affection between them. Use one paragraph of about 160-210 words.'),
('D01','diagnostic','physics','unconstrained explanation','brief','Why does a soap bubble show changing colors? Give a brief explanation for an interested adult.'),
('D02','diagnostic','mind/behavior','ambiguity handling','90-130','Someone says, "The experiment proved that listening to music makes everyone concentrate better." Explain what would need to be known before accepting this claim. Use one paragraph of about 90-130 words.'),
('D03','diagnostic','mathematics','reasoning and correction','90-130','A puzzle says that doubling the side length of a square doubles its area. Explain the mistake and give a clear correction for a younger reader. Write one paragraph of about 90-130 words.'),
('D04','diagnostic','arts/language','analogy','160-210','Explain why revising a musical performance is different from revising a printed score. Use an analogy of your choice. Write one paragraph of about 160-210 words.'),
('D05','diagnostic','computing','format freedom','brief','What is lost and what is gained when a computer stores a photograph at lower resolution? Explain briefly using whatever structure seems clearest.'),
('D06','diagnostic','philosophy','controlled contrast','90-130','A person keeps a promise for the wrong reason, while another breaks one to prevent serious harm. Compare how intention and consequence affect our judgment. Write one paragraph of about 90-130 words without prescribing a final verdict.')
]
bank=[dict(zip(['prompt_id','split','domain','genre','requested_words','prompt_text'],x),bank_version='1.0',origin='independently authored for this research',prompt_family_id=x[0],template_family_id=x[0],wrapper_id='paragraph-length-v1' if x[4]!='brief' else 'brief-natural-v1') for x in prompts]
(OUT/'collection_prompt_bank.json').write_text(json.dumps(bank,ensure_ascii=False,indent=2))
intro='''NAME THAT LLM - MANUAL COLLECTION PROMPT BANK
Research cutoff: 2 November 2026. Version 1.0.

Paste ONE prompt into ONE fresh conversation. Do not paste this entire file.
Give the same prompt to every accessible exact nominal target model. Never
claim a newer model, an Auto router, or a confirmed alias redirect is the old
target. gemini-3.7-flash currently redirects to 3.8: leave that cell missing.

N01-N24: initial matched prose training panel.
H01-H06: untouched external evaluation; never use for acquisition or tuning.
D01-D06: separate diagnostic experiment, not default training mixture.

The six H prompts are a small transfer sentinel, not enough to establish
small performance differences. Expand with approximately 30 or more independent
holdout families before relying on accuracy comparisons; report uncertainty.
The core panel balances four length bands for coverage, not to reproduce
the empirical training length histogram. Use the global histogram to guide
larger collections, with identical length cells across models.

The bank deliberately spans length and subject. For a first matched panel,
use the text exactly as written across models. Do not reroll merely because
an answer is too long, too short, unusual, incorrect, or stylistically weak.
Record actual length and task compliance. Later use several preregistered
wording templates, crossed across models, to test dependence on this wrapper.

These are independent new prompts, not recovered competition prompts.
They are designed hypotheses, not an empirically optimized collection bank.
The complete protocol and metadata schema are in the research report/kit.
'''
(OUT/'Name_That_LLM_Collection_Prompts.txt').write_text(intro+'\n\n'+'\n\n'.join(f"{p['prompt_id']} | {p['split']} | {p['domain']} | {p['requested_words']} words\n{p['prompt_text']}" for p in bank))

nullable_string={'type':['string','null']}
schema={
 '$schema':'https://json-schema.org/draft/2020-12/schema',
 'title':'Name That LLM collection record, version 1.0','type':'object','additionalProperties':False,
 'required':['record_id','schema_version','target_label','served_model_id','provider','interface','collected_at_utc','prompt_id','prompt_family_id','template_family_id','prompt_bank_version','prompt_text','raw_response','source_verification','session','settings','provenance','quality','split'],
 'properties':{
  'record_id':{'type':'string'},'schema_version':{'const':'1.0'},'target_label':nullable_string,
  'served_model_id':nullable_string,'displayed_model_label':nullable_string,'snapshot_id':nullable_string,
  'provider':{'type':'string'},'interface':{'type':'string'},'interface_version':nullable_string,
  'endpoint_or_product_url':nullable_string,'region':nullable_string,'collected_at_utc':{'type':'string','format':'date-time'},
  'prompt_id':{'type':'string'},'prompt_family_id':{'type':'string'},'template_family_id':{'type':'string'},'wrapper_id':nullable_string,'prompt_bank_version':{'type':'string'},'prompt_text':{'type':'string'},
  'raw_response':{'type':'string'},'rendered_text_export':nullable_string,'raw_response_sha256':nullable_string,
  'visible_reasoning':nullable_string,'final_answer_span':nullable_string,
  'source_verification':{'type':'object','required':['status','evidence','routing_uncertainty','competition_snapshot_match'],
   'properties':{'status':{'enum':['snapshot_verified','nominal_target_verified','ui_label_verified','family_proxy','rejected_exact_label']},'evidence':{'type':'array','items':{'type':'string'}},'routing_uncertainty':{'type':'string'},'competition_snapshot_match':{'enum':['verified','unknown','mismatch']},'verification_date':nullable_string}},
  'session':{'type':'object','required':['fresh_conversation','prior_turn_count','memory','personalization','custom_instructions','system_instructions_status','tools'],
   'properties':{'fresh_conversation':{'type':'boolean'},'prior_turn_count':{'type':'integer','minimum':0},'session_id_hash':nullable_string,'memory':{'enum':['off','on','unknown']},'personalization':{'enum':['off','on','unknown']},'custom_instructions':{'enum':['empty','present_recorded','unknown']},'system_instructions_status':{'enum':['known_recorded','provider_default_unknown','partially_known']},'known_system_instructions':nullable_string,'tools':{'type':'array','items':{'type':'string'}},'model_fallback_observed':{'type':['boolean','null']}}},
  'settings':{'type':'object','required':['temperature','top_p','seed','max_output_tokens','reasoning_effort'],
   'properties':{'temperature':{'type':['number','null']},'top_p':{'type':['number','null']},'seed':{'type':['integer','null']},'max_output_tokens':{'type':['integer','null']},'reasoning_effort':nullable_string,'settings_source':nullable_string}},
  'provenance':{'type':'object','required':['collection_method','copy_method','collector_id','license_or_terms_url','eligibility_status','transforms','attempt_index','parent_record_id'],
   'properties':{'collection_method':{'enum':['manual_chat','manual_playground','api','public_corpus']},'copy_method':{'type':'string'},'collector_id':{'type':'string'},'original_artifact_reference':nullable_string,'license_or_terms_url':nullable_string,'source_dataset_url':nullable_string,'eligibility_status':{'enum':['pending_clarification','organizer_confirmed','public_source_reviewed','not_eligible']},'eligibility_evidence':nullable_string,'transforms':{'type':'array','items':{'type':'string'}},'attempt_index':{'type':'integer','minimum':1},'parent_record_id':nullable_string,'request_id':nullable_string}},
  'quality':{'type':'object','required':['word_count','complete','refusal','tool_use_observed','duplicate_of','issues'],
   'properties':{'word_count':{'type':'integer','minimum':0},'tokenizer_probe':nullable_string,'token_count':{'type':['integer','null']},'complete':{'type':'boolean'},'refusal':{'type':'boolean'},'tool_use_observed':{'type':'boolean'},'duplicate_of':nullable_string,'issues':{'type':'array','items':{'type':'string'}}}},
  'cost':{'type':'object','properties':{'input_tokens':{'type':['integer','null']},'output_tokens':{'type':['integer','null']},'reasoning_tokens':{'type':['integer','null']},'usd':{'type':['number','null']},'operator_minutes':{'type':['number','null']}}},
  'split':{'enum':['train','external_holdout','diagnostic','quarantine']}
 }}
(OUT/'collection_record.schema.json').write_text(json.dumps(schema,indent=2))
example={
 'record_id':'example-template-do-not-train','schema_version':'1.0','target_label':None,'served_model_id':None,'displayed_model_label':None,'snapshot_id':None,
 'provider':'FILL_ME','interface':'FILL_ME','collected_at_utc':'2026-11-02T00:00:00Z','prompt_id':'N01','prompt_family_id':'N01','template_family_id':'N01','wrapper_id':'paragraph-length-v1','prompt_bank_version':'1.0','prompt_text':bank[0]['prompt_text'],'raw_response':'REPLACE WITH UNEDITED RESPONSE','rendered_text_export':None,'raw_response_sha256':None,
 'source_verification':{'status':'family_proxy','evidence':[],'routing_uncertainty':'not verified','competition_snapshot_match':'unknown'},
 'session':{'fresh_conversation':True,'prior_turn_count':0,'memory':'unknown','personalization':'unknown','custom_instructions':'unknown','system_instructions_status':'provider_default_unknown','tools':[]},
 'settings':{'temperature':None,'top_p':None,'seed':None,'max_output_tokens':None,'reasoning_effort':None},
 'provenance':{'collection_method':'manual_chat','copy_method':'FILL_ME','collector_id':'FILL_ME','license_or_terms_url':None,'eligibility_status':'pending_clarification','transforms':[],'attempt_index':1,'parent_record_id':None},
 'quality':{'word_count':0,'complete':False,'refusal':False,'tool_use_observed':False,'duplicate_of':None,'issues':['TEMPLATE ONLY']},'split':'quarantine'}
(OUT/'collection_record_template.json').write_text(json.dumps(example,ensure_ascii=False,indent=2))
print('Wrote 36 independent prompts (24 training, 6 holdout, 6 diagnostic), schema, and empty template.')
