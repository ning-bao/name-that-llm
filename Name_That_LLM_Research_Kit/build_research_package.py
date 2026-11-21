"""Render a research report and bundle its reproducible, nontraining kit."""
from pathlib import Path
import csv,hashlib,html,json,re,shutil,zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'deliverables';OUT.mkdir(exist_ok=True)
S=json.loads((ROOT/'forensics/forensic_summary.json').read_text())
T=pd.read_csv(ROOT/'forensics/model_statistics.csv').set_index('model')
TOP=json.loads((ROOT/'topic_audit.json').read_text())
BANK=json.loads((OUT/'collection_prompt_bank.json').read_text())

ORDER=['gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra','gpt-6-astra','gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash','claude-fable-5','claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5','qwen3.8-max','glm-5.3','glm-5.3-flash','kimi-k3']

def mdtable(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(c) for c in row)+' |' for row in rows])

modeltab=mdtable(['Exact competition label','Mean words','Mean probe tokens','Em dash %','Curly apostrophe %'],[[m,f'{T.loc[m,"words_mean"]:.1f}',f'{T.loc[m,"tokens_mean"]:.1f}',f'{T.loc[m,"has_em_dash_pct"]:.1f}',f'{T.loc[m,"has_curly_apostrophe_pct"]:.1f}'] for m in ORDER])
domainnames={'space_astronomy':'Space and astronomy','mathematics_computing':'Mathematics and computing','life_sciences':'Life sciences','society_economics':'Society and economics','history_culture_arts_language':'History, culture, arts and language','philosophy_ethics':'Philosophy and ethics','earth_environment':'Earth and environment','physics':'Physics','mind_behavior':'Mind and behavior'}
domaintab=mdtable(['Inferred domain','Prompt groups'],[[domainnames[k],v] for k,v in sorted(TOP['domain_counts'].items(),key=lambda x:-x[1])]+[['Total',150]])

access=[
 ['gpt-5.5','API documented; chat retired','API; verify snapshot',5,30,'R17,R33'],
 ['gpt-5.6-luna','API documented','Work/Codex; not ordinary chat selector',.2,1.2,'R18,R33'],
 ['gpt-5.6-sol','API documented','Chat/Work/Codex; record effort',4,20,'R18,R33'],
 ['gpt-5.6-terra','API documented','Work/Codex; not ordinary chat selector',2,12,'R18,R33'],
 ['gpt-6-astra','API documented','GPT-6 Pro; plan-dependent',10,50,'R18,R33'],
 ['gemini-3.1-pro-preview','API preview documented','AI Studio; not generic Pro label',2,12,'R20'],
 ['gemini-3.6-flash','API documented','AI Studio; explicit ID',.75,3.75,'R20'],
 ['gemini-3.7-flash','Redirected to 3.8','No verified current exact route',None,None,'R19'],
 ['gemini-3.8-flash','API documented','AI Studio; explicit ID',.75,3.75,'R20'],
 ['claude-fable-5','API active legacy','Current consumer selector unverified',10,50,'R21,R22'],
 ['claude-opus-4-6','API active legacy','Current consumer selector unverified',5,25,'R21,R22'],
 ['claude-opus-4-8','API active legacy','Current consumer selector unverified',5,25,'R21,R22'],
 ['claude-opus-5','API active legacy','Current consumer selector unverified',5,25,'R21,R22'],
 ['claude-sonnet-5','API active legacy','Current consumer selector unverified',2,10,'R21,R22'],
 ['qwen3.8-max','API documented','Chat route unverified; regional rates',1.65,4.951,'R23'],
 ['glm-5.3','API documented','Consumer chat exact route unverified',1.4,4.4,'R24'],
 ['glm-5.3-flash','API documented','Do not substitute FlashX',.15,.5,'R24'],
 ['kimi-k3','API documented','Explicit K3; avoid Cluster mode',3,15,'R25']]
with (OUT/'model_access_and_prices.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['target_label','documented_status','manual_route','usd_per_million_input','usd_per_million_output','references']);w.writerows(access)
accesstab=mdtable(['Exact target','API / manual access','Input / output $ per 1M'],[[m,status+'. '+route,('Not exact 3.7' if i is None else f'{i:g} / {o:g}')+' ['+refs+']'] for m,status,route,i,o,refs in access])

refs=[
 ('D1','Supplied train.csv, test.csv and sample_submission.parquet. Full read-only programmatic audit; forensic_summary.json, model_statistics.csv, controlled_effects.csv and forensic_analysis.py in the kit. Token probe: o200k_base; whitespace word counts; no attribution model fit.',[]),
 ('D2','Manual semantic audit of every prompt group. topic_audit.json records inferred domains, genres, inspection scope, ambiguity and topical overlaps. These are not original prompt texts.',[]),
 ('R01','Kaggle. Name That LLM: overview, evaluation and notebook requirements. Official competition source.',['https://www.kaggle.com/competitions/name-that-llm/overview/description']),
 ('R02','Kaggle. Name That LLM: dataset description and schema. Official competition source.',['https://www.kaggle.com/competitions/name-that-llm/data']),
 ('R03','Official competition rules supplied by the owner: Pasted markdown(20261102-060939).md, 35,906 bytes. Entire attachment read. Live rules URL was not independently readable; section references in this report refer to the attachment.',['https://www.kaggle.com/competitions/name-that-llm/rules']),
 ('R04','Modarres et al. Every Model Leaves a Trace: Multi-Resolution LLM Fingerprinting. 20 October 2026 preprint, v1; prompt-grouped experiments and Table 2.',['https://arxiv.org/abs/2610.28434','https://arxiv.org/html/2610.28434v1']),
 ('R05','Guo et al. Profiler: Black-box AI-generated Text Origin Detection via Context-aware Inference Pattern Analysis. EMNLP 2025 proceedings paper.',['https://aclanthology.org/2025.emnlp-main.1265/']),
 ('R06','La Cava and Tagarelli. OpenTuringBench: An Open-Model-based Benchmark and Framework for Machine-Generated Text Detection and Attribution. EMNLP 2025, Tables 2-3.',['https://aclanthology.org/2025.emnlp-main.1354/','https://aclanthology.org/2025.emnlp-main.1354.pdf']),
 ('R07','READER: Dynamic LLM Provenance from Query-Varying Interactions. 2026 preprint, v3; Agent500 protocol and appendix Table 8.',['https://arxiv.org/abs/2606.10794','https://arxiv.org/html/2606.10794v3']),
 ('R08','Ko et al. Who\'s Behind the Text? Robust Model Authorship Attribution (ATLAS). 14 October 2026 preprint, v1; five-family MAABench and source-conditioned adapters.',['https://arxiv.org/abs/2610.19583','https://arxiv.org/html/2610.19583v1']),
 ('R09','Wang et al. SemEval-2024 Task 8: Multidomain, Multimodel and Multilingual Machine-Generated Text Detection. Official shared-task report, Table 11.',['https://aclanthology.org/2024.semeval-1.279/','https://aclanthology.org/2024.semeval-1.279.pdf']),
 ('R10','Chen et al. Team MGTD4ADL at SemEval-2024 Task 8. Proceedings system paper; enriched-test augmentation comparison in Table 5.',['https://aclanthology.org/2024.semeval-1.245/','https://aclanthology.org/2024.semeval-1.245.pdf']),
 ('R11','Hu et al. Fingerprinting LLMs via Prompt Injection. ACL 2026 main proceedings paper; LLMPrint.',['https://aclanthology.org/2026.acl-long.541/','https://github.com/hifi-hyp/ACL-LLMPrint']),
 ('R12','Pasquini et al. LLMmap: Fingerprinting for Large Language Models. USENIX Security 2025. Active model querying.',['https://www.usenix.org/conference/usenixsecurity25/presentation/pasquini']),
 ('R13','Kurian, Holland and Oesch. Attacks & Defenses Against LLM Fingerprinting. August 2025 preprint, v1. RL query selection on a fixed query pool.',['https://arxiv.org/html/2508.09021v1']),
 ('R14','Ettin: an Open Suite of Paired Encoders and Decoders. ICLR 2026; official JHU model card and MIT license.',['https://arxiv.org/abs/2507.11412','https://huggingface.co/jhu-clsp/ettin-encoder-400m']),
 ('R15','DMIS Lab. UltraBERT: One Encoder for Classification and Retrieval. Author release 14 October 2026 and preprint; checkpoint and dependency documentation.',['https://huggingface.co/blog/dmis-lab/ultrabert','https://arxiv.org/abs/2610.15028']),
 ('R16','Qwen. Qwen3-Embedding-8B official checkpoint, Apache 2.0 model card.',['https://huggingface.co/Qwen/Qwen3-Embedding-8B']),
 ('R17','OpenAI. Model Release Notes; October 14 GPT-5.5 chat retirement does not affect API access.',['https://help.openai.com/en/articles/9624314-model-release-notes']),
 ('R18','OpenAI. GPT-6 and other models in ChatGPT; exact product/model availability.',['https://help.openai.com/en/articles/20001354-gpt-6-and-other-models-in-chatgpt']),
 ('R19','Google. Gemini API release notes and deprecations; October 8 Gemini 3.7-to-3.8 automatic routing.',['https://ai.google.dev/gemini-api/docs/changelog','https://ai.google.dev/gemini-api/docs/deprecations/']),
 ('R20','Google. Current Gemini API pricing; Flash promotional prices and output/thinking billing.',['https://ai.google.dev/gemini-api/docs/pricing']),
 ('R21','Anthropic. Model deprecations and lifecycle status.',['https://platform.claude.com/docs/en/about-claude/model-deprecations']),
 ('R22','Anthropic. Official target-model pages and rates; current Sonnet 5 rate used rather than an older overview.',['https://platform.claude.com/docs/en/models/fable-5/overview','https://platform.claude.com/docs/en/models/opus-4-6/overview','https://platform.claude.com/docs/en/models/opus-4-8/overview','https://platform.claude.com/docs/en/models/opus-5/overview','https://platform.claude.com/docs/en/models/sonnet-5/overview']),
 ('R23','Alibaba Cloud. qwen3.8-max official model identifier and regional pricing. Singapore/international $2/$6 differs from some global regions.',['https://www.alibabacloud.com/help/en/model-studio/qwen3-8-max']),
 ('R24','Z.AI. Official pricing and target model documentation.',['https://docs.z.ai/guides/overview/pricing','https://docs.z.ai/guides/llm/glm-5.3','https://docs.z.ai/guides/vlm/glm-5.3-flash']),
 ('R25','Moonshot AI. Kimi K3 official announcement/API price and model-mode selection.',['https://www.kimi.com/en/blog/kimi-k3','https://www.kimi.com/en/help/others/model-mode-selection']),
 ('R26','NVIDIA. L40S official specification: 48 GB and BF16 support.',['https://www.nvidia.com/en-sg/data-center/l40s/']),
 ('R27','Hugging Face. PEFT quantization documentation, version 0.20.0; NF4 and BF16 configuration.',['https://huggingface.co/docs/peft/v0.20.0/developer_guides/quantization']),
 ('R28','Scikit-learn. TfidfVectorizer implementation; character analyzers normalize whitespace internally.',['https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/feature_extraction/text.py']),
 ('R29','Zhu et al. Learned Sparse Lexical Representations for Large-scale Authorship Attribution. Findings of EMNLP 2026; human-authorship evidence, indirect for this task.',['https://aclanthology.org/2026.findings-emnlp.400/']),
 ('R30','Google DeepMind. SynthID and text watermarking. Mechanism exists, but applicability to these records/versions is not established.',['https://deepmind.google/models/synthid/','https://deepmind.google/blog/watermarking-ai-generated-text-and-video-with-synthid/']),
 ('R31','Kaggle participant discussion: Does the test set keep the one-passage-per-model-per-prompt structure? Unverified self-reported CV/leaderboard experiment, not organizer evidence.',['https://www.kaggle.com/competitions/name-that-llm/discussion/751802']),
 ('R32','Maniceet Sahay. Name That LLM: OpenRouter + OpenAI Generations. Public dataset description retrieved via search; underlying rows not audited in this report.',['https://www.kaggle.com/maniceet/name-that-llm-openrouter-openai-generations']),
 ('R33','OpenAI. Official API model identifiers and current input/output rates.',['https://developers.openai.com/api/docs/models/gpt-5.5','https://developers.openai.com/api/docs/models/gpt-5.6-luna','https://developers.openai.com/api/docs/models/gpt-5.6-sol','https://developers.openai.com/api/docs/models/gpt-5.6-terra','https://developers.openai.com/api/docs/models/gpt-6-astra']),
 ('R34','Beigi et al. Supervised contrastive domain-generalization study for machine-generated text attribution; DSAA 2024. Three older models and fake-news setting.',['https://arxiv.org/abs/2407.21264']),
 ('R35','Profiler official implementation. Per-source binary detection and ROC-AUC evaluation.',['https://github.com/MarkGHX/Profiler']),
 ('R36','ATLAS official implementation. Source-conditioned likelihood modeling.',['https://github.com/komseok0109/ATLAS']),
 ('R37','READER official implementation. Response-only mode, trajectory features and evidence accumulation; MIT license.',['https://github.com/LeoJeshua/READER']),
 ('R38','OpenTuringBench official dataset card. CC BY-NC 4.0; assess rights separately before incorporating data.',['https://huggingface.co/datasets/MLNTeam-Unical/OpenTuringBench'])
]
reftext='\n\n'.join('['+k+'] '+d+(' '+' '.join(urls) if urls else '') for k,d,urls in refs)
(OUT/'references.json').write_text(json.dumps([{'id':k,'description':d,'urls':u} for k,d,u in refs],indent=2))
prompttext='\n\n'.join('### '+p['prompt_id']+' | '+p['split']+' | '+p['domain']+'\n\n'+p['prompt_text'] for p in BANK)

report=(ROOT/'report_source.txt').read_text().replace('[[MODEL_STATS]]',modeltab).replace('[[DOMAIN_STATS]]',domaintab).replace('[[ACCESS_TABLE]]',accesstab).replace('[[PROMPT_BANK]]',prompttext).replace('[[REFERENCES]]',reftext)
plain=report.replace('[[FINGERPRINT_FIGURE]]','Figure: Name_That_LLM_Fingerprints.png (included separately).')
(OUT/'Name_That_LLM_Research_Report_2026-11-02.txt').write_text(plain)

# Figure: all exact labels, percentage of passages containing each punctuation.
cols=['has_em_dash_pct','has_curly_apostrophe_pct','has_straight_apostrophe_pct','has_colon_pct','has_semicolon_pct']
vals=T.loc[ORDER,cols].to_numpy()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig,ax=plt.subplots(figsize=(10.8,9.5))
fig.patch.set_facecolor('white')
im=ax.imshow(vals,cmap='Blues',vmin=0,vmax=100,aspect='auto')
ax.set_xticks(range(len(cols)),['Em dash','Curly\napostrophe','Straight\napostrophe','Colon','Semicolon'])
ax.set_yticks(range(18),ORDER)
ax.tick_params(length=0,pad=8)
ax.xaxis.tick_top()
for i in range(18):
 for j in range(5):ax.text(j,i,f'{vals[i,j]:.1f}',ha='center',va='center',color='white' if vals[i,j]>62 else '#18364b',fontsize=9.5)
for boundary in [4.5,8.5,13.5,14.5,16.5]:ax.axhline(boundary,color='white',linewidth=3)
for spine in ax.spines.values():spine.set_visible(False)
fig.suptitle('Some punctuation differences are much larger across families',x=.05,y=.98,ha='left',fontsize=16,fontweight='bold',color='#17374d')
fig.text(.05,.925,'Measured on 2,699 supplied training passages. Values are percentages containing the mark.',fontsize=10,color='#425466')
fig.text(.05,.035,'Each label has 150 passages except claude-fable-5 (149). These are descriptive training statistics,\nnot validation accuracy or deterministic rules for individual passages.',fontsize=9,color='#425466')
fig.subplots_adjust(left=.30,right=.97,top=.86,bottom=.10)
figure=OUT/'Name_That_LLM_Fingerprints.png';fig.savefig(figure,dpi=190,bbox_inches='tight');plt.close(fig)

# Reproducible planned split manifest. This is not model training.
groups=sorted(pd.read_csv(ROOT/'inputs/train.csv').prompt_id.unique())
def hsort(items,prefix):return sorted(items,key=lambda x:hashlib.sha256((prefix+x).encode()).hexdigest())
confirm=hsort(groups,'name-that-llm-confirm-v1|')[:30];dev=[x for x in groups if x not in confirm]
plan={'version':'1.0','status':'post-EDA, before any attribution training; design manifest only','confirmation_prompts':confirm,'development_prompts':dev,'initialization_seeds':[17,43],'repetitions':[]}
for rep in [0,1]:
 ordered=hsort(dev,f'name-that-llm-dev-v1-rep{rep}|');folds=[]
 for outer in range(5):
  valid=ordered[outer*24:(outer+1)*24];train=[p for p in ordered if p not in valid]
  inner=hsort(train,f'name-that-llm-inner-v1-rep{rep}-outer{outer}|')
  folds.append({'outer_fold':outer,'train':train,'validation':valid,'inner_validation':[inner[j*32:(j+1)*32] for j in range(3)]})
 plan['repetitions'].append({'partition_id':rep,'folds':folds})
(OUT/'validation_plan.json').write_text(json.dumps(plan,indent=2))

# Polished PDF with repeated table headers and continuous page numbers.
fontdir=Path('/usr/share/fonts/truetype/dejavu')
for name,file in [('DV','DejaVuSans.ttf'),('DV-Bold','DejaVuSans-Bold.ttf'),('DV-Italic','DejaVuSans.ttf')]:pdfmetrics.registerFont(TTFont(name,str(fontdir/file)))
pdfmetrics.registerFontFamily('DV',normal='DV',bold='DV-Bold',italic='DV-Italic',boldItalic='DV-Bold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyN',fontName='DV',fontSize=9.3,leading=13.3,textColor=colors.HexColor('#21313e'),spaceAfter=7.2))
styles.add(ParagraphStyle(name='TitleN',fontName='DV-Bold',fontSize=25,leading=31,textColor=colors.HexColor('#16384d'),spaceAfter=18,keepWithNext=True))
styles.add(ParagraphStyle(name='H1N',fontName='DV-Bold',fontSize=17,leading=22,textColor=colors.HexColor('#16384d'),spaceBefore=18,spaceAfter=11,keepWithNext=True))
styles.add(ParagraphStyle(name='H2N',fontName='DV-Bold',fontSize=11.4,leading=16,textColor=colors.HexColor('#167d8d'),spaceBefore=11,spaceAfter=7,keepWithNext=True))
styles.add(ParagraphStyle(name='PromptN',fontName='DV-Bold',fontSize=10.6,leading=14,textColor=colors.HexColor('#167d8d'),spaceBefore=9,spaceAfter=5,keepWithNext=True))
styles.add(ParagraphStyle(name='CellN',fontName='DV',fontSize=7.5,leading=10.4,textColor=colors.HexColor('#21313e'),spaceAfter=0))
styles.add(ParagraphStyle(name='CellHeadN',fontName='DV-Bold',fontSize=7.7,leading=10.4,textColor=colors.white,spaceAfter=0))
styles.add(ParagraphStyle(name='RefN',fontName='DV',fontSize=8.1,leading=11.5,textColor=colors.HexColor('#31424e'),spaceAfter=8))
styles.add(ParagraphStyle(name='BulletN',parent=styles['BodyN'],leftIndent=10,firstLineIndent=-8))

def esc(s):
 s=html.escape(s)
 # Links remain human-readable; long words may wrap in Paragraph.
 s=re.sub(r'(https?://[^\s<>]+)',lambda m:'<link href="'+m.group(1)+'" color="#176c9b">'+m.group(1)+'</link>',s)
 return s

W,H=A4;MARGIN=44;WIDTH=W-2*MARGIN
def make_table(lines):
 rows=[[c.strip() for c in line.strip().strip('|').split('|')] for line in lines]
 rows=[row for row in rows if not all(re.fullmatch(r':?-+:?',x) for x in row)]
 n=len(rows[0]);weights=[1]*n
 first=rows[0][0]
 if first=='Exact competition label':weights=[2.6,1,1.25,1,1.55]
 elif first=='Exact target':weights=[1.6,2.5,1.5]
 elif first=='Rank':weights=[.42,1.25,2,1.8,1.5,1.1]
 elif first=='Order':weights=[.45,1.8,1.5,1.25,2]
 elif first=='Method':weights=[1.35,1.4,1.5,1.35,1.6]
 elif n==2:weights=[1.2,3.4] if first=='Field group' else [3,1]
 widths=[WIDTH*w/sum(weights) for w in weights]
 processed=[[Paragraph(esc(c),styles['CellHeadN'] if i==0 else styles['CellN']) for c in row] for i,row in enumerate(rows)]
 table=Table(processed,colWidths=widths,repeatRows=1,hAlign='LEFT')
 commands=[('BACKGROUND',(0,0),(-1,0),colors.HexColor('#17384d')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.8,colors.HexColor('#167d8d'))]
 for i in range(1,len(rows)):
  if i%2==0:commands.append(('BACKGROUND',(0,i),(-1,i),colors.HexColor('#eef3f6')))
  commands.append(('LINEBELOW',(0,i),(-1,i),.25,colors.HexColor('#d4dfe6')))
 table.setStyle(TableStyle(commands))
 return KeepTogether([table]) if first=='Exact competition label' else table

story=[];lines=report.splitlines();i=0;inrefs=False
while i<len(lines):
 line=lines[i].strip()
 if not line:i+=1;continue
 if line=='[[FINGERPRINT_FIGURE]]':
  story.append(Image(str(figure),width=WIDTH,height=WIDTH*9.5/10.8));story.append(Spacer(1,9));i+=1;continue
 if line.startswith('|'):
  block=[]
  while i<len(lines) and lines[i].strip().startswith('|'):block.append(lines[i]);i+=1
  story.append(make_table(block));story.append(Spacer(1,10));continue
 if line.startswith('# '):story.append(Paragraph(esc(line[2:]),styles['TitleN']));i+=1;continue
 if line.startswith('## '):
  title=line[3:];inrefs=title.startswith('J.')
  if re.match(r'^[B-J]\. ',title):story.append(PageBreak())
  story.append(Paragraph(esc(title),styles['H1N']));i+=1;continue
 if line.startswith('### '):
  heading=line[4:];style=styles['PromptN'] if re.match(r'^[NHD]\d{2} \|',heading) else styles['H2N']
  story.append(Paragraph(esc(heading),style));i+=1;continue
 block=[line];i+=1
 while i<len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(('#','|','[[FINGERPRINT')) and not re.match(r'^(?:- |\d+\. )',lines[i].strip()):
  block.append(lines[i].strip());i+=1
 text=' '.join(block)
 style=styles['RefN'] if inrefs and re.match(r'^\[(?:R|D)\d',text) else styles['BulletN'] if re.match(r'^(?:- |\d+\. )',text) else styles['BodyN']
 story.append(Paragraph(esc(text),style))

def page(canvas,doc):
 canvas.saveState();canvas.setStrokeColor(colors.HexColor('#d9e2e8'));canvas.setLineWidth(.5)
 canvas.line(MARGIN,H-29,W-MARGIN,H-29)
 canvas.setFont('DV',7.4);canvas.setFillColor(colors.HexColor('#526878'))
 canvas.drawString(MARGIN,H-21,'NAME THAT LLM  /  INDEPENDENT RESEARCH')
 canvas.drawRightString(W-MARGIN,H-21,'2 NOVEMBER 2026')
 canvas.line(MARGIN,31,W-MARGIN,31)
 canvas.drawString(MARGIN,20,'Research and experimental planning - no training or submissions performed')
 canvas.drawRightString(W-MARGIN,20,str(doc.page))
 canvas.restoreState()

pdf=OUT/'Name_That_LLM_Research_Report_2026-11-02.pdf'
doc=SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=MARGIN,leftMargin=MARGIN,topMargin=46,bottomMargin=43,title='Name That LLM: independent research and experiment plan',author='Independent research',pageCompression=1)
doc.build(story,onFirstPage=page,onLaterPages=page)

# Derived data and source scripts only; raw competition records are not bundled.
kit=OUT/'kit';kit.mkdir(exist_ok=True)
for name in ['forensic_summary.json','model_statistics.csv','family_statistics.csv','controlled_effects.csv','near_duplicate_candidates.csv']:
 shutil.copy2(ROOT/'forensics'/name,kit/name)
shutil.copy2(ROOT/'topic_audit.json',kit/'topic_audit.json')
for name in ['forensic_analysis.py','build_collection_kit.py','build_research_package.py','report_source.txt']:shutil.copy2(ROOT/name,kit/name)
for name in ['collection_prompt_bank.json','collection_record.schema.json','collection_record_template.json','validation_plan.json','model_access_and_prices.csv','references.json','Name_That_LLM_Collection_Prompts.txt','Name_That_LLM_Research_Report_2026-11-02.txt','Name_That_LLM_Fingerprints.png']:shutil.copy2(OUT/name,kit/name)
shutil.copy2(pdf,kit/pdf.name)
(kit/'README.txt').write_text('''NAME THAT LLM RESEARCH KIT - 2 NOVEMBER 2026

Start with the PDF or complete text research report. The prompt file contains
24 core training prompts, six external holdout sentinels and six diagnostics.
Use one prompt per fresh conversation, and verify source identity first.
The JSON schema and template define a collection record; the template is not
training data. No source-model calls, classifier training or predictions have
been made. Raw competition passages are deliberately not included.

Reproduce read-only numeric forensics with:
  python forensic_analysis.py --input PATH_TO_SUPPLIED_FILES --output audit
Dependencies: pandas, numpy, scipy, scikit-learn, pyarrow, tiktoken.
Input directory must contain train.csv, test.csv and sample_submission.parquet.
The first tokenizer use may download its public o200k_base vocabulary.
The analysis never edits those inputs and does not fit an attribution model.

The feature statistics use whitespace word counts, ASCII lexical regexes,
heuristic sentence/LaTeX markers and one common tokenizer probe. Controlled
R-squared values are descriptive in-sample effects, not validation accuracy.
topic_audit.json contains manual inferred annotations, not recovered prompts.
The nearest-cross-prompt list is a selected nearest-neighbor diagnostic, not
an exhaustive ranking. Full threshold and maximum-similarity checks are given.

validation_plan.json freezes a POST-EDA 30-prompt confirmation set and nested
five-outer/three-inner development folds on the other 120 prompts. It also
contains a second development partition. No model errors have been used.
All broad EDA already used the full corpus; do not call this a pristine test.

Prices/access are documentation-based at the cutoff, not authenticated access
tests. Paid/private external-data eligibility and exact historical snapshots
remain conditional. gemini-3.7-flash currently redirects to 3.8 in Google's API.

To regenerate the report, place these files in the layout expected by the
build scripts: forensics/ for statistic files, inputs/ for original supplied
files, topic_audit.json and report_source.txt at root; run build_collection_kit.py
then build_research_package.py. PDF dependencies: reportlab, matplotlib and
DejaVu fonts. Build scripts perform no model training or collection.
''')
manifest={str(p.relative_to(kit)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(kit.iterdir()) if p.is_file() and p.name!='artifact_manifest.json'}
(kit/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
zip_path=OUT/'Name_That_LLM_Research_Kit.zip'
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in sorted(kit.iterdir()):
  if p.is_file():z.write(p,arcname=p.name)
print(json.dumps({'pdf':str(pdf),'zip':str(zip_path),'word_count':len(plain.split()),'files':len(list(kit.iterdir())),'image':str(figure)},indent=2))
