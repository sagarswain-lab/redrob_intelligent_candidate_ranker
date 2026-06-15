"""
reference_data.py

Lookup tables for how much each skill, job title, company, and location
should count toward a candidate's score, plus the job description text used
for the TF-IDF comparison. All of these numbers came from going through the
full 100K-candidate file first (see notebooks/01_eda_full_pool.py) - skills
and titles are bucketed by how common they are and how relevant they look
for an AI engineering role, companies by whether they're AI-native startups,
FAANG, well-known product companies, or the consulting firms the job
description specifically rules out.

See README.md for the full breakdown of each table and why the numbers land
where they do.
"""

# ---------------------------------------------------------------------------
# SKILL WEIGHTS
# ---------------------------------------------------------------------------
# weight scale: 0 (irrelevant) .. 14 (ultra-rare "ideal candidate" marker)

SKILL_WEIGHTS = {}

# --- Tier 1: general/business/SWE skills, ~12% prevalence -> weight 0 -------
_TIER1 = [
    "HTML","Databricks","Redux","Terraform","Angular","Figma","Salesforce CRM",
    "Vue.js","Sales","Accounting","Agile","Kafka","Excel","BigQuery","CI/CD",
    "Project Management","Airflow","AWS","Flask","Scrum","Illustrator",
    "Kubernetes","ETL","CSS","Docker","Next.js","Apache Beam","Java","Go",
    "TypeScript","JavaScript","dbt","REST APIs","Spark","Marketing","Tally",
    "GraphQL","Snowflake","Webpack","Six Sigma","SEO","SAP","GCP","PostgreSQL",
    "Rust","Apache Flink","gRPC","Content Writing","SQL","Hadoop","Redis",
    "Tailwind","Photoshop","FastAPI","Microservices","PowerPoint","Spring Boot",
    "Data Pipelines","Django","MongoDB","Node.js","Azure","React",
]
for _s in _TIER1:
    SKILL_WEIGHTS[_s] = 0

# --- Tier 2: "buzzword AI", ~5% prevalence -----------------------------------
# 2a. Core retrieval/embedding buzzwords - relevant but common/stuffable
for _s in ["Embeddings","Vector Search","Semantic Search","Sentence Transformers",
           "RAG","Information Retrieval","Recommendation Systems","Pinecone","FAISS"]:
    SKILL_WEIGHTS[_s] = 5

# 2b. LLM / prompt-framework skills - nice-to-have, but LangChain-heavy
#     profiles are explicitly the "framework enthusiast" warning in the JD
for _s in ["LLMs","Prompt Engineering","Hugging Face Transformers","Fine-tuning LLMs"]:
    SKILL_WEIGHTS[_s] = 3
SKILL_WEIGHTS["LangChain"] = 2  # explicitly the "framework tutorial" red flag

# 2c. CV / speech - JD explicitly says CV/speech-without-NLP is not a fit
for _s in ["YOLO","GANs","OpenCV","ASR","Image Classification","Computer Vision",
           "Speech Recognition","CNN","Diffusion Models","TTS","Object Detection"]:
    SKILL_WEIGHTS[_s] = 1

# 2d. General ML/MLOps - broadly useful, moderate
for _s in ["Feature Engineering","Data Science","Reinforcement Learning","MLOps",
           "BentoML","Kubeflow","MLflow","Weights & Biases","Time Series",
           "Forecasting","Statistical Modeling"]:
    SKILL_WEIGHTS[_s] = 2

# --- Tier 3: "core ML/IR practitioner" skills, ~1.3% prevalence -------------
# Named vector DBs / hybrid search infra (JD must-have, named explicitly)
for _s in ["Qdrant","Weaviate","Milvus","pgvector","OpenSearch","Elasticsearch"]:
    SKILL_WEIGHTS[_s] = 9
# Ranking + evaluation (closest skill-level proxy to JD's NDCG/MRR/MAP ask)
for _s in ["BM25","Learning to Rank"]:
    SKILL_WEIGHTS[_s] = 10
# RAG frameworks (production retrieval)
for _s in ["Haystack","LlamaIndex"]:
    SKILL_WEIGHTS[_s] = 8
# Core foundations - "Strong Python" + ML/DL/NLP fundamentals (JD must-have)
SKILL_WEIGHTS["Python"] = 9
SKILL_WEIGHTS["NLP"] = 8
SKILL_WEIGHTS["PyTorch"] = 7
SKILL_WEIGHTS["TensorFlow"] = 6
SKILL_WEIGHTS["scikit-learn"] = 6
SKILL_WEIGHTS["Machine Learning"] = 6
SKILL_WEIGHTS["Deep Learning"] = 6
# Fine-tuning (JD nice-to-have)
for _s in ["LoRA","QLoRA","PEFT"]:
    SKILL_WEIGHTS[_s] = 6

# --- Tier 4: ultra-rare IR-specific skills, 1-7 occurrences in 100K ---------
# These belong almost exclusively to the designed "Tier 5 ideal" candidates.
for _s in ["Information Retrieval Systems","Search Backend","Text Encoders",
           "Vector Representations","Content Matching","Model Adaptation",
           "Ranking Systems","Search & Discovery","Workflow Orchestration",
           "Search Infrastructure","Indexing Algorithms","Document Processing",
           "Natural Language Processing"]:
    SKILL_WEIGHTS[_s] = 13
# "Open-source ML libraries" -> maps to JD's "external validation" ask
SKILL_WEIGHTS["Open-source ML libraries"] = 14

DEFAULT_SKILL_WEIGHT = 0  # any skill name not listed above


# ---------------------------------------------------------------------------
# TITLE TIERS
# ---------------------------------------------------------------------------
# 5 bands across the 47 distinct current_title values in the dataset.

TITLE_TIER_SCORE = {}

# Band 1: off-target, non-tech roles (~69K candidates total, ~5,500-5,800 each)
# The JD is explicit: "a candidate who has all the AI keywords listed as
# skills but whose title is 'Marketing Manager' is not a fit, no matter how
# perfect their skill list looks." -> hard penalty.
for _t in ["Business Analyst","HR Manager","Mechanical Engineer","Accountant",
           "Project Manager","Customer Support","Operations Manager",
           "Content Writer","Sales Executive","Civil Engineer",
           "Graphic Designer","Marketing Manager"]:
    TITLE_TIER_SCORE[_t] = -20

# Band 2: general tech, non-ML (~25.7K candidates total, ~2,700-3,500 each)
for _t in ["Software Engineer","Full Stack Developer","Cloud Engineer",
           "Java Developer",".NET Developer","DevOps Engineer",
           "Mobile Developer","Frontend Engineer","QA Engineer"]:
    TITLE_TIER_SCORE[_t] = 0

# Band 3: data-adjacent (~4.3K candidates total, ~650-770 each)
for _t in ["Analytics Engineer","Data Engineer","Data Analyst",
           "Backend Engineer","Senior Data Engineer","Senior Software Engineer"]:
    TITLE_TIER_SCORE[_t] = 3

# Band 4: AI/ML "broad" (~1,000 candidates total, ~130-170 each)
for _t in ["ML Engineer","AI Research Engineer","Data Scientist",
           "Senior Software Engineer (ML)","Computer Vision Engineer",
           "Junior ML Engineer","AI Specialist"]:
    TITLE_TIER_SCORE[_t] = 8

# Band 5: AI/ML "narrow/senior" (~179 candidates total, 3-26 each)
for _t in ["Recommendation Systems Engineer","Machine Learning Engineer",
           "Applied ML Engineer","Search Engineer","AI Engineer",
           "Senior Data Scientist","NLP Engineer","Senior NLP Engineer",
           "Senior Machine Learning Engineer","Staff Machine Learning Engineer",
           "Senior AI Engineer","Senior Applied Scientist","Lead AI Engineer"]:
    TITLE_TIER_SCORE[_t] = 12

DEFAULT_TITLE_SCORE = -5  # any title not in the 47 known values


# ---------------------------------------------------------------------------
# COMPANY TIERS
# ---------------------------------------------------------------------------
# product_exposure_score: how much this company says "real product engineering"
# Used as: max(score) over {current_company} U {all career_history companies}

AI_NATIVE_COMPANIES = {
    "Sarvam AI","Krutrim","Genpact AI","Glance","Rephrase.ai","Aganitha",
    "Niramai","Saarthi.ai","Mad Street Den","Observe.AI","Wysa","Haptik",
    "Verloop.io","Yellow.ai","Locobuzz",
}
FAANG_COMPANIES = {
    "Google","Meta","Amazon","Microsoft","Apple","Netflix","LinkedIn",
    "Salesforce","Adobe","Uber",
}
INDIAN_PRODUCT_COMPANIES = {
    "Razorpay","CRED","Flipkart","Zomato","Swiggy","Ola","PhonePe","Paytm",
    "Meesho","Nykaa","InMobi","Zoho","Freshworks","Dream11","PolicyBazaar",
    "BYJU'S","Vedantu","Unacademy","upGrad","PharmEasy",
}
# JD's explicit "people who have ONLY worked at consulting firms" disqualifier
CONSULTING_COMPANIES = {
    "TCS","Infosys","Wipro","Accenture","Capgemini","Cognizant",
    "Tech Mahindra","Mphasis","HCL","Mindtree",
}
# Generic pop-culture filler companies used as neutral filler across all
# title types - carry no signal either way.
FILLER_COMPANIES = {
    "Wayne Enterprises","Initech","Pied Piper","Globex Inc","Acme Corp",
    "Dunder Mifflin","Hooli","Stark Industries",
}

COMPANY_PRODUCT_SCORE = {}
for _c in AI_NATIVE_COMPANIES:
    COMPANY_PRODUCT_SCORE[_c] = 8
for _c in FAANG_COMPANIES:
    COMPANY_PRODUCT_SCORE[_c] = 7
for _c in INDIAN_PRODUCT_COMPANIES:
    COMPANY_PRODUCT_SCORE[_c] = 5
for _c in CONSULTING_COMPANIES:
    COMPANY_PRODUCT_SCORE[_c] = 0
for _c in FILLER_COMPANIES:
    COMPANY_PRODUCT_SCORE[_c] = 0

DEFAULT_COMPANY_SCORE = 0


# ---------------------------------------------------------------------------
# LOCATION TIERS
# ---------------------------------------------------------------------------
# JD: "Pune/Noida-preferred... Candidates in Hyderabad, Pune, Mumbai, Delhi
# NCR welcome... Outside India: case-by-case, but we don't sponsor visas."

LOCATION_TOP = {"noida", "pune"}                       # explicitly preferred
LOCATION_WELCOME = {                                    # explicitly welcomed
    "delhi", "new delhi", "gurgaon", "gurugram",
    "faridabad", "ghaziabad", "mumbai", "hyderabad",
}
# everything else in India falls through to the relocation-dependent score


# ---------------------------------------------------------------------------
# JOB DESCRIPTION REFERENCE TEXT (for TF-IDF lexical/semantic similarity)
# ---------------------------------------------------------------------------
# A condensed version of job_description.md, focused on the requirements /
# "ideal candidate" sections. Used as the query document for TF-IDF cosine
# similarity against each candidate's (summary + career descriptions + skill
# names). This is the "lexical retrieval" half of the hybrid score.

JD_TEXT = """
Senior AI Engineer, founding AI engineering team at an AI-native talent
intelligence platform. Own the intelligence layer: ranking, retrieval, and
matching systems that decide what recruiters see when they search for
candidates and what candidates see when they search for roles. Ship a v2
ranking system involving embeddings, hybrid retrieval, vector search, and
possibly LLM-based re-ranking on top of an existing BM25 and rule-based
scoring system. Set up offline evaluation benchmarks (NDCG, MRR, MAP),
online A/B testing, and recruiter-feedback loops.

Must have production experience with embeddings-based retrieval systems
(sentence-transformers, OpenAI embeddings, BGE, E5) deployed to real users,
including embedding drift, index refresh, and retrieval-quality regression.
Must have production experience with vector databases or hybrid search
infrastructure: Pinecone, Weaviate, Qdrant, Milvus, OpenSearch,
Elasticsearch, or FAISS. Strong Python and code quality. Hands-on experience
designing evaluation frameworks for ranking systems: NDCG, MRR, MAP,
offline-to-online correlation, A/B test interpretation.

Nice to have: LLM fine-tuning experience with LoRA, QLoRA, or PEFT;
learning-to-rank models such as XGBoost-based or neural rankers; prior
exposure to HR-tech, recruiting tech, or marketplace products; background in
distributed systems or large-scale inference optimization; open-source
contributions in the AI/ML space, papers, or talks as external validation.

Do not want: pure research/academic profiles with no production deployment;
candidates whose AI experience is only recent (<12 months) LangChain +
OpenAI projects without pre-LLM-era ML production experience; senior
engineers who have not written production code in 18 months because they
moved into pure architecture or tech-lead roles; title-chasers who hop
companies every 1.5 years chasing Senior to Staff to Principal titles;
framework enthusiasts whose portfolio is LangChain tutorials and "how I
built X with hot framework Y" blog posts rather than systems thinking;
candidates whose entire career has been at consulting firms such as TCS,
Infosys, Wipro, Accenture, Cognizant, Capgemini, Tech Mahindra, Mphasis, HCL,
or Mindtree with no product company experience; candidates whose primary
expertise is computer vision, speech, or robotics without significant
NLP or information-retrieval exposure; candidates whose work has been
entirely on closed-source proprietary systems for 5+ years with no external
validation such as papers, talks, or open source.

Experience: 5 to 9 years, ideally 6 to 8 years total, with 4 to 5 years in
applied ML or AI roles at product companies, not pure services companies.
Has shipped at least one end-to-end ranking, search, recommendation, or
retrieval system to real users at meaningful scale. Has strong informed
opinions about hybrid versus dense retrieval, offline versus online
evaluation, and when to fine-tune versus prompt an LLM, grounded in systems
they actually built. Located in or willing to relocate to Noida or Pune,
India; candidates in Hyderabad, Mumbai, or Delhi NCR are also welcome.
Active in the job market or otherwise reachable, with a short notice period
ideally under 30 days.
""".strip()


# ---------------------------------------------------------------------------
# Retrieval-related skill names (any tier) - used for the reasoning
# generator's "has at least one embeddings/vector-DB/retrieval/ranking
# skill" check, kept separate from SKILL_WEIGHTS so the check is exact
# (name-based) rather than threshold-based.
# ---------------------------------------------------------------------------
RETRIEVAL_RELATED_SKILLS = {
    "Embeddings", "Vector Search", "Semantic Search", "Sentence Transformers",
    "RAG", "Information Retrieval", "Recommendation Systems", "Pinecone", "FAISS",
    "Qdrant", "Weaviate", "Milvus", "pgvector", "OpenSearch", "Elasticsearch",
    "BM25", "Learning to Rank", "Haystack", "LlamaIndex",
    "Information Retrieval Systems", "Search Backend", "Text Encoders",
    "Vector Representations", "Content Matching", "Ranking Systems",
    "Search & Discovery", "Search Infrastructure", "Indexing Algorithms",
}


# ---------------------------------------------------------------------------
# JD-CONNECTION PHRASES
# ---------------------------------------------------------------------------
# Maps a candidate's top-weighted skill to a short clause explaining *which*
# part of the JD it satisfies. Used as the reasoning's closing sentence when
# a candidate has no flagged concerns, so even "clean" profiles get an
# explicit JD-connection rather than a generic "good fit" filler. Each group
# has 2 phrasing variants for sentence-level variety.

SKILL_GROUP = {}
for _s in ["Qdrant", "Weaviate", "Milvus", "pgvector", "OpenSearch", "Elasticsearch",
           "Pinecone", "FAISS"]:
    SKILL_GROUP[_s] = "vector_db"
for _s in ["BM25", "Learning to Rank"]:
    SKILL_GROUP[_s] = "ranking_eval"
for _s in ["Embeddings", "Vector Search", "Semantic Search", "Sentence Transformers",
           "RAG", "Information Retrieval", "Recommendation Systems"]:
    SKILL_GROUP[_s] = "embeddings"
for _s in ["Haystack", "LlamaIndex"]:
    SKILL_GROUP[_s] = "rag_framework"
for _s in ["Python", "NLP", "Machine Learning", "Deep Learning", "PyTorch",
           "TensorFlow", "scikit-learn"]:
    SKILL_GROUP[_s] = "core_ml"
for _s in ["LoRA", "QLoRA", "PEFT"]:
    SKILL_GROUP[_s] = "finetuning"
SKILL_GROUP["Open-source ML libraries"] = "oss"
for _s in ["Information Retrieval Systems", "Search Backend", "Text Encoders",
           "Vector Representations", "Content Matching", "Model Adaptation",
           "Ranking Systems", "Search & Discovery", "Workflow Orchestration",
           "Search Infrastructure", "Indexing Algorithms", "Document Processing",
           "Natural Language Processing"]:
    SKILL_GROUP[_s] = "systems_depth"

GROUP_PHRASES = {
    "vector_db": [
        "is exactly the production vector-DB / hybrid-search experience the JD asks for",
        "lines up directly with the JD's named vector-database requirement",
    ],
    "ranking_eval": [
        "matches the JD's focus on ranking-evaluation work (the NDCG/MRR/MAP side of the role)",
        "is the kind of ranking and evaluation background the role is hiring for",
    ],
    "embeddings": [
        "is a direct match for the JD's embeddings-based retrieval requirement",
        "covers the core retrieval and recommendation work the JD describes",
    ],
    "rag_framework": [
        "shows hands-on production RAG-framework experience, as the JD asks for",
        "is directly relevant to the retrieval-pipeline work the role centers on",
    ],
    "core_ml": [
        "covers the strong-Python / core-ML foundation the JD lists as a must-have",
        "is the kind of ML engineering foundation the role is built on",
    ],
    "finetuning": [
        "covers the LLM fine-tuning nice-to-have the JD calls out",
        "is a useful match for the JD's fine-tuning nice-to-have",
    ],
    "oss": [
        "provides exactly the external-validation signal (open-source contributions) the JD calls out",
        "speaks to the open-source / external-validation angle the JD mentions",
    ],
    "systems_depth": [
        "is precisely the search and ranking-systems depth this role is built around",
        "is a near-exact match for the founding AI engineer role's day-to-day work",
    ],
}

DEFAULT_JD_PHRASES = [
    "broadly fits the AI/ML focus of the role",
    "is relevant background for the retrieval and ranking work the role centers on",
]
