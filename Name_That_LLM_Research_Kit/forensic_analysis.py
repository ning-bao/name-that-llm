"""Read-only corpus forensics. No classifier fitting, training, or prediction.

Usage: python forensic_analysis.py --input INPUT_DIR --output OUTPUT_DIR
Requires pandas, numpy, scipy, scikit-learn, pyarrow, tiktoken.
Statistics use whitespace word counts and o200k_base tokens as a common probe,
not any candidate source model's native tokenizer. Raw files are never edited.
"""
import argparse, collections, hashlib, json, math, re, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import tiktoken

ap=argparse.ArgumentParser()
ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
args=ap.parse_args();src=Path(args.input);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(src/'train.csv',keep_default_na=False)
test=pd.read_csv(src/'test.csv',keep_default_na=False)
sub=pd.read_parquet(src/'sample_submission.parquet')
enc=tiktoken.get_encoding('o200k_base')

def family(m):
    for p,n in [('gpt','OpenAI'),('gemini','Google'),('claude','Anthropic'),('qwen','Qwen'),('glm','ZAI'),('kimi','Moonshot')]:
        if m.startswith(p):return n
    raise ValueError(m)

punct={'em_dash':'—','en_dash':'–','curly_apostrophe':'’','left_curly_quote':'“','right_curly_quote':'”',
       'straight_apostrophe':"'",'straight_double_quote':'"','semicolon':';','colon':':',
       'exclamation':'!','question':'?','asterisk':'*','backtick':'`','slash':'/','newline':'\n','tab':'\t','nbsp':'\u00a0'}
rows=[]
for s in df.text:
    words=s.split();lex=re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?",s)
    sentence_ends=re.findall(r'[.!?]+(?:[\"\u201d\u2019]*)\s+|[.!?]+$',s)
    r={'words':len(words),'tokens':len(enc.encode(s,disallowed_special=())), 'chars':len(s),'bytes':len(s.encode()),
       'lex_words':len(lex),'type_token_ratio':len(set(x.lower() for x in lex))/max(1,len(lex)),
       'mean_word_chars':sum(map(len,lex))/max(1,len(lex)),
       'sentences_heuristic':len(sentence_ends),'paragraphs':len(re.split(r'\n\s*\n',s)),
       'digit_per_1k':sum(c.isdigit() for c in s)*1000/max(1,len(s)),
       'nonascii_per_1k':sum(ord(c)>127 for c in s)*1000/max(1,len(s)),
       'uppercase_per_1k':sum(c.isupper() for c in s)*1000/max(1,len(s)),
       'leading_space':int(bool(s) and s[0].isspace()),'trailing_space':int(bool(s) and s[-1].isspace()),
       'repeated_spaces':int(bool(re.search(r' {2,}',s))), 'blankline':int('\n\n' in s),
       'has_bold':int('**' in s),'has_heading':int(bool(re.search(r'(?m)^\s*#{1,6}\s',s))),
       'has_bullet':int(bool(re.search(r'(?m)^\s*[-*•]\s',s))),
       'has_numbered':int(bool(re.search(r'(?m)^\s*\d+[.)]\s',s))),
       'has_latex':int(bool(re.search(r'\\[([]|\\(?:frac|sum|sqrt|times|alpha|beta|infty|math)|\$[^$\n]+\$',s))),
       'has_code_fence':int('```' in s), 'has_url':int(bool(re.search(r'https?://|www\.',s))),
       'nfc_changed':int(unicodedata.normalize('NFC',s)!=s),
       'nfkc_changed':int(unicodedata.normalize('NFKC',s)!=s),
       'start_certainly':int(bool(re.match(r'(?i)\s*certainly\b',s))),
       'start_sure':int(bool(re.match(r'(?i)\s*sure\b',s))),
       'second_person_per_100w':len(re.findall(r'(?i)\b(?:you|your|yours)\b',s))*100/max(1,len(words)),
       'first_person_per_100w':len(re.findall(r'(?i)\b(?:I|me|my|we|our|us)\b',s))*100/max(1,len(words)),
       'negation_per_100w':len(re.findall(r'(?i)\b(?:not|never|no)\b|n[’\']t\b',s))*100/max(1,len(words))}
    for name,p in punct.items():
        r['has_'+name]=int(p in s);r[name+'_per_1k']=s.count(p)*1000/max(1,len(s))
    rows.append(r)
features=pd.DataFrame(rows)
data=pd.concat([df[['id','prompt_id','model']],features],axis=1);data['family']=df.model.map(family)
data.to_csv(out/'row_features.csv',index=False)

summary={
 'files':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in src.iterdir() if p.is_file()},
 'train_shape':list(df.shape),'train_columns':df.columns.tolist(),'test_shape':list(test.shape),'test_columns':test.columns.tolist(),
 'submission_shape':list(sub.shape),'submission_columns':sub.columns.tolist(),'submission_dtypes':{k:str(v) for k,v in sub.dtypes.items()},
 'submission_ids_match':bool(set(sub.id)==set(test.id)),
 'empty_fields':(df=='').sum().to_dict(),'model_counts':df.model.value_counts().sort_index().to_dict(),
 'prompt_counts':df.groupby('prompt_id').size().value_counts().to_dict(),
 'incomplete_prompts':{k:sorted(set(df.model)-set(g.model)) for k,g in df.groupby('prompt_id') if len(g)!=18},
 'id_duplicates':int(df.id.duplicated().sum()),'prompt_model_duplicates':int(df.duplicated(['prompt_id','model']).sum()),
 'exact_text_duplicates':int(df.text.duplicated().sum()),'train_test_id_overlap':len(set(df.id)&set(test.id)),
 'train_test_exact_text_overlap':len(set(df.text)&set(test.text)),
 'whitespace_lowercase_text_duplicates':int(df.text.str.replace(r'\s+',' ',regex=True).str.strip().str.lower().duplicated().sum()),
 'length_overall':features[['words','tokens','chars','sentences_heuristic','paragraphs']].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).to_dict(),
 'words_outside_40_250':int(((features.words<40)|(features.words>250)).sum()),
 'token_gt_512':int((features.tokens>512).sum()),'token_gt_768':int((features.tokens>768).sum()),
 'artifact_totals':{c:int(features[c].sum()) for c in features if c.startswith('has_') or c in ['leading_space','trailing_space','repeated_spaces','blankline','nfc_changed','nfkc_changed']},
 'non_ascii_char_counts':dict(collections.Counter(c for s in df.text for c in s if ord(c)>127).most_common(35)),
 'note':'All lengths measured; sentence counts heuristic. No models trained. Token probe o200k_base, not source-specific.'
}
cols=['words','tokens','chars','sentences_heuristic','paragraphs','type_token_ratio','mean_word_chars']
per_model=data.groupby('model')[cols].agg(['mean','median','min','max','std'])
per_model.columns=['_'.join(x) for x in per_model.columns]
for c in ['words','tokens']:
    for q in [.05,.95,.99]:per_model[c+'_p'+str(round(q*100))]=data.groupby('model')[c].quantile(q)
for c in features.columns:
    if c.startswith('has_'):per_model[c+'_pct']=data.groupby('model')[c].mean()*100
per_model.to_csv(out/'model_statistics.csv')
data.groupby('family')[['has_'+p for p in punct]+['has_latex','has_bold','has_bullet','has_numbered','words','tokens']].mean().to_csv(out/'family_statistics.csv')

# Descriptive prompt/length-controlled variance: two-way fixed-effects projection.
# This is statistical variance decomposition, not a fitted attribution classifier.
G=pd.get_dummies(df.prompt_id,dtype=float).to_numpy()
M=pd.get_dummies(df.model,dtype=float).to_numpy()[:,1:]
F=pd.get_dummies(data.family,dtype=float).to_numpy()[:,1:]
logw=np.log1p(features.words.to_numpy());logw=(logw-logw.mean())/logw.std()
X0=np.column_stack([G,logw,logw**2])
featcols=[c for c in features if c.endswith('_per_1k') or c.endswith('_per_100w') or c in ['type_token_ratio','mean_word_chars','sentences_heuristic','paragraphs','tokens']]
Y=features[featcols].to_numpy(dtype=float)
R0=Y-X0@lstsq(X0,Y)[0]
XF=np.column_stack([X0,F]);RF=Y-XF@lstsq(XF,Y)[0]
XM=np.column_stack([X0,M]);RM=Y-XM@lstsq(XM,Y)[0]
ss0=(R0**2).sum(0);ssf=(RF**2).sum(0);ssm=(RM**2).sum(0)
partial=pd.DataFrame({'feature':featcols,'family_partial_r2':np.divide(ss0-ssf,ss0,out=np.zeros_like(ss0),where=ss0>1e-10),'model_partial_r2':np.divide(ss0-ssm,ss0,out=np.zeros_like(ss0),where=ss0>1e-10),'within_family_incremental_r2':np.divide(ssf-ssm,ssf,out=np.zeros_like(ssf),where=ssf>1e-10)}).sort_values('model_partial_r2',ascending=False)
partial.to_csv(out/'controlled_effects.csv',index=False)

# Crossed design length variation, fitted additively for descriptive effect size.
W=features[['words','tokens']].to_numpy(dtype=float);WG=W-G@lstsq(G,W)[0]
WGM=W-np.column_stack([G,M])@lstsq(np.column_stack([G,M]),W)[0]
summary['length_model_partial_r2_after_prompt']={c:float(1-(WGM[:,i]**2).sum()/(WG[:,i]**2).sum()) for i,c in enumerate(['words','tokens'])}
summary['length_prompt_r2']={c:float(1-(WG[:,i]**2).sum()/((W[:,i]-W[:,i].mean())**2).sum()) for i,c in enumerate(['words','tokens'])}

# Same-prompt versus cross-prompt similarity and near-duplicate candidates.
vec=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,lowercase=False,max_features=180000,dtype=np.float32)
V=vec.fit_transform(df.text);S=(V@V.T).toarray();np.fill_diagonal(S,-1)
pid=df.prompt_id.to_numpy();same=pid[:,None]==pid[None,:]
si=np.where(np.triu(same,1));di=np.where(np.triu(~same,1))
summary['cosine_similarity']={'definition':'case-sensitive sklearn char3-5 TFIDF; min_df2 max_features180000; repeated whitespace normalized by sklearn internally; descriptive only',
    'same_prompt_percentiles':np.quantile(S[si],[.5,.9,.95,.99,1]).tolist(),
    'cross_prompt_percentiles':np.quantile(S[di],[.5,.9,.95,.99,1]).tolist()}
pairs=[]
for i,j in zip(*np.where(np.triu(S>=.80,1))):pairs.append({'id_a':df.id[i],'id_b':df.id[j],'prompt_a':pid[i],'prompt_b':pid[j],'model_a':df.model[i],'model_b':df.model[j],'cosine':float(S[i,j])})
pd.DataFrame(pairs,columns=['id_a','id_b','prompt_a','prompt_b','model_a','model_b','cosine']).sort_values('cosine',ascending=False).to_csv(out/'near_duplicate_candidates.csv',index=False)
summary['near_duplicate_candidates_ge080']=len(pairs)
summary['cross_prompt_candidates_ge080']=sum(x['prompt_a']!=x['prompt_b'] for x in pairs)
bestcross=np.where(~same,S,-1).argmax(1)
cross_sorted=sorted([{'id_a':df.id[i],'id_b':df.id[int(j)],'prompt_a':pid[i],'prompt_b':pid[int(j)],'cosine':float(S[i,j])} for i,j in enumerate(bestcross) if i<int(j)],key=lambda x:-x['cosine'])[:30]
summary['selected_nearest_cross_prompt_pairs']=cross_sorted

# Phrase frequencies require occurrence in >=6 distinct prompts; do not report
# passage text. Association is descriptive and needs held-out confirmation.
cv=CountVectorizer(ngram_range=(2,4),lowercase=True,binary=True,min_df=6,max_features=60000,token_pattern=r"(?u)\b\w[\w'’]*\b")
A=cv.fit_transform(df.text);terms=cv.get_feature_names_out();nlabels=df.model.nunique()
global_count=np.asarray(A.sum(axis=0)).ravel();gp=G.T@A;prompt_count=np.asarray((gp>0).sum(0)).ravel()
phrases={}
for m in sorted(df.model.unique()):
    ii=(df.model==m).to_numpy();cnt=np.asarray(A[ii].sum(axis=0)).ravel();other=global_count-cnt
    score=np.log((cnt+.5)/(ii.sum()+1))-np.log((other+.5)/(len(df)-ii.sum()+1))
    keep=(cnt>=6)&(prompt_count>=6);ix=np.where(keep)[0];ix=ix[np.argsort(score[ix])[::-1]][:10]
    phrases[m]=[{'phrase':terms[j],'own_rows':int(cnt[j]),'other_rows':int(other[j]),'distinct_prompts':int(prompt_count[j]),'log_rate_ratio':float(score[j])} for j in ix]
summary['repeated_phrases']=phrases

# How many rows are altered by common preprocessing policies?
summary['preprocessing_changed_rows']={
 'strip':sum(s.strip()!=s for s in df.text),
 'collapse_whitespace':sum(re.sub(r'\s+',' ',s)!=s for s in df.text),
 'lowercase':sum(s.lower()!=s for s in df.text),
 'ascii_ignore':sum(s.encode('ascii','ignore').decode()!=s for s in df.text),
 'smart_punctuation_fold':sum(s.translate(str.maketrans({'’':"'",'‘':"'",'“':'"','”':'"','—':'-','–':'-'}))!=s for s in df.text)}

(out/'forensic_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ['repeated_phrases','selected_nearest_cross_prompt_pairs','files']},ensure_ascii=False,indent=2))
print('\nPER MODEL CORE\n'+per_model[['words_mean','words_median','words_min','words_max','tokens_mean','tokens_p95','tokens_max','has_em_dash_pct','has_curly_apostrophe_pct','has_latex_pct']].round(2).to_string())
print('\nCONTROLLED EFFECTS\n'+partial.head(18).round(4).to_string(index=False))
