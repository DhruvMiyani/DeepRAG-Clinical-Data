# DeepRAG Clinical Data Pipeline Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          DeepRAG Clinical Question Answering System             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                    USER QUERY                                    │
│                     "What are risk factors for pressure injuries?"               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
```

## Core Architecture Flow

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                  DEEPRAG PIPELINE                                     │
│                                                                                        │
│  ┌────────────────┐        ┌─────────────────┐        ┌──────────────────┐          │
│  │   User Query   │───────▶│  DeepRAG Core   │───────▶│  MDP Decision    │          │
│  │                │        │   (deeprag_     │        │    Framework     │          │
│  │                │        │    core.py)     │        │                  │          │
│  └────────────────┘        └─────────────────┘        └──────────────────┘          │
│                                      │                          │                     │
│                                      ▼                          ▼                     │
│                            ┌─────────────────┐        ┌──────────────────┐          │
│                            │  Binary Tree    │        │   Atomic         │          │
│                            │    Search       │◀──────▶│   Decisions      │          │
│                            │                 │        │                  │          │
│                            └─────────────────┘        └──────────────────┘          │
│                                      │                                                │
│                    ┌─────────────────┴─────────────────┐                            │
│                    ▼                                   ▼                             │
│         ┌──────────────────┐                ┌──────────────────┐                    │
│         │    Retrieval     │                │   Parametric     │                    │
│         │      Path        │                │   Knowledge      │                    │
│         └──────────────────┘                └──────────────────┘                    │
│                    │                                   │                             │
│                    ▼                                   ▼                             │
│         ┌──────────────────┐                ┌──────────────────┐                    │
│         │  Vector Store    │                │    GPT-5 LLM     │                    │
│         │    (FAISS)       │                │   Generation     │                    │
│         └──────────────────┘                └──────────────────┘                    │
│                    │                                   │                             │
│                    └───────────────┬───────────────────┘                             │
│                                    ▼                                                 │
│                         ┌──────────────────┐                                        │
│                         │  Chain of        │                                        │
│                         │  Calibration     │                                        │
│                         └──────────────────┘                                        │
│                                    │                                                 │
│                                    ▼                                                 │
│                         ┌──────────────────┐                                        │
│                         │   Final Answer   │                                        │
│                         └──────────────────┘                                        │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                               MIMIC-III DATA INTEGRATION                              │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                ┌───────────────────────────┼───────────────────────────┐
                ▼                           ▼                           ▼
    ┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
    │   HAPI Dataset   │        │  HAAKI Dataset   │        │   HAA Dataset    │
    │  (Pressure       │        │  (Kidney         │        │   (Anemia)       │
    │   Injuries)      │        │   Injury)        │        │                  │
    └──────────────────┘        └──────────────────┘        └──────────────────┘
                │                           │                           │
                └───────────────────────────┼───────────────────────────┘
                                            ▼
                              ┌──────────────────────┐
                              │  NosocomialData     │
                              │     Loader          │
                              │  (datasets.py)      │
                              └──────────────────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    ▼                       ▼                       ▼
          ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
          │  Chronologies   │    │   Admissions    │    │     Labels      │
          │  - timestamps   │    │  - admit times  │    │   - outcomes    │
          │  - observations │    │  - patient IDs  │    │   - diagnoses   │
          └─────────────────┘    └─────────────────┘    └─────────────────┘
                    │                       │                       │
                    └───────────────────────┼───────────────────────┘
                                            ▼
                              ┌──────────────────────┐
                              │  MimicDeepRAG       │
                              │    Integrator       │
                              │ (mimic_deeprag_     │
                              │  integration.py)    │
                              └──────────────────────┘
                                            │
                                            ▼
                              ┌──────────────────────┐
                              │  Document Creation  │
                              │  - Clinical texts   │
                              │  - Metadata         │
                              └──────────────────────┘
                                            │
                                            ▼
                              ┌──────────────────────┐
                              │   Text Splitter     │
                              │  - Chunk size: 750  │
                              │  - Overlap: 100     │
                              └──────────────────────┘
                                            │
                                            ▼
                              ┌──────────────────────┐
                              │  OpenAI Embeddings  │
                              │  text-embedding-    │
                              │     ada-002         │
                              └──────────────────────┘
                                            │
                                            ▼
                              ┌──────────────────────┐
                              │   FAISS Vector      │
                              │      Store          │
                              └──────────────────────┘
```

## Component Details

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              MODULE ARCHITECTURE                                 │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  📁 Core Modules                                                                  │
│  ├── 📄 config.py ──────────────────► Environment Configuration                  │
│  │                                    - API Keys                                  │
│  │                                    - Model Settings                            │
│  │                                                                                │
│  ├── 📄 deeprag_pipeline.py ────────► Main Pipeline Orchestrator                 │
│  │   │                                - Question Processing                       │
│  │   │                                - Response Generation                       │
│  │   │                                                                            │
│  │   └──► 📄 deeprag_core.py ───────► DeepRAG Core Logic                        │
│  │       │                            - MDP Framework                             │
│  │       │                            - Binary Tree Search                        │
│  │       │                            - Atomic Decisions                          │
│  │       │                                                                        │
│  │       └──► 📄 deeprag_training.py ► Training Components                       │
│  │                                    - Imitation Learning                        │
│  │                                    - Chain of Calibration                      │
│  │                                                                                │
│  ├── 📄 datasets.py ────────────────► MIMIC-III Data Loader                     │
│  │   │                                - CSV File Reading                          │
│  │   │                                - Data Structuring                          │
│  │   │                                                                            │
│  │   └──► 📄 mimic_deeprag_        ► Integration Layer                          │
│  │         integration.py             - Document Conversion                       │
│  │                                    - Clinical Knowledge                        │
│  │                                    - Vector Store Creation                     │
│  │                                                                                │
│  └── 📄 test_mimic_integration.py ──► Testing & Validation                      │
│                                       - Sample Data Processing                    │
│                                       - Clinical Q&A Testing                      │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

## Query Processing Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            QUERY PROCESSING PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────────────┘

1. USER QUERY INPUT
   │
   ▼
   "What are the risk factors for hospital-acquired pressure injuries?"
   │
   ▼
2. DEEPRAG DECISION TREE
   │
   ├─► RETRIEVE? ─────► YES ─────► VECTOR SEARCH
   │                               │
   │                               ▼
   │                               Similarity Search in FAISS
   │                               │
   │                               ▼
   │                               Top-K Documents (k=6)
   │                               │
   │                               ▼
   │                               Context Assembly
   │
   └─► PARAMETRIC? ───► YES ─────► Direct LLM Generation
                                   │
                                   ▼
                                   GPT-5 Response
   
3. RETRIEVAL NARRATIVE GENERATION
   │
   ▼
   Subqueries:
   - "pressure injury risk factors"
   - "HAPI prevention"
   - "Braden scale assessment"
   │
   ▼
4. CONTEXT AGGREGATION
   │
   ▼
   Retrieved Documents:
   - Patient chronologies with C0392747 codes
   - Clinical knowledge about pressure injuries
   - Risk assessment protocols
   │
   ▼
5. FINAL ANSWER GENERATION
   │
   ▼
   GPT-5 synthesizes comprehensive answer with:
   - Clinical evidence from MIMIC-III
   - Medical knowledge
   - Specific risk factors
   │
   ▼
6. CHAIN OF CALIBRATION
   │
   ▼
   Quality check and refinement
   │
   ▼
7. RESPONSE TO USER
```

## Data Schema

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               MIMIC-III DATA SCHEMA                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  CHRONOLOGIES TABLE                                                              │
│  ┌──────────────┬─────────────┬──────────────┬─────────────────────────┐       │
│  │ subject_id   │   hadm_id   │  timestamp   │     observations        │       │
│  ├──────────────┼─────────────┼──────────────┼─────────────────────────┤       │
│  │     17       │   161087    │  2015-04-30  │ C0392747 C1248432 ...  │       │
│  │     21       │   145834    │  2015-05-01  │ C0008902 C1553702 ...  │       │
│  └──────────────┴─────────────┴──────────────┴─────────────────────────┘       │
│                                                                                   │
│  ADMISSIONS TABLE                                                                │
│  ┌──────────────┬─────────────┬────────────────────────┐                       │
│  │ subject_id   │   hadm_id   │      admittime         │                       │
│  ├──────────────┼─────────────┼────────────────────────┤                       │
│  │     17       │   161087    │  2015-04-28 14:30:00   │                       │
│  │     21       │   145834    │  2015-04-29 09:15:00   │                       │
│  └──────────────┴─────────────┴────────────────────────┘                       │
│                                                                                   │
│  LABELS TABLE                                                                    │
│  ┌──────────────┬─────────────┬──────────────┬─────────────┐                   │
│  │ subject_id   │   hadm_id   │   condition  │    label    │                   │
│  ├──────────────┼─────────────┼──────────────┼─────────────┤                   │
│  │     17       │   161087    │     HAPI     │      1      │                   │
│  │     21       │   145834    │     HAPI     │      0      │                   │
│  └──────────────┴─────────────┴──────────────┴─────────────┘                   │
│                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Technology Stack

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TECHNOLOGY STACK                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  🧠 AI/ML Components                                                             │
│  ├── OpenAI GPT-5 ─────────► Large Language Model                               │
│  ├── OpenAI Embeddings ────► text-embedding-ada-002                             │
│  └── FAISS ────────────────► Facebook AI Similarity Search                      │
│                                                                                   │
│  🔧 Frameworks                                                                   │
│  ├── LangChain ────────────► LLM Application Framework                          │
│  ├── Pandas ───────────────► Data Processing                                    │
│  └── NumPy ────────────────► Numerical Computing                                │
│                                                                                   │
│  📊 Data Sources                                                                 │
│  ├── MIMIC-III ────────────► Clinical Database                                  │
│  ├── HAPI Dataset ─────────► Pressure Injury Data                               │
│  ├── HAAKI Dataset ────────► Kidney Injury Data                                 │
│  └── HAA Dataset ──────────► Anemia Data                                        │
│                                                                                   │
│  🔐 Configuration                                                                │
│  ├── Environment Variables ► .env file                                          │
│  └── API Keys ─────────────► OpenAI API                                         │
│                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```