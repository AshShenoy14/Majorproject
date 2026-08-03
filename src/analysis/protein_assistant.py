"""
ProteinAssistant: An intelligent knowledge-based assistant for protein biology.
Uses curated knowledge + project data to answer common questions about proteins,
diseases, drug targets, and the TransGraph-PPI system.
"""

import re
import random
import os
from typing import Optional, Dict, List
from google import genai
from dotenv import load_dotenv

# Load environment variables (for GEMINI_API_KEY)
load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    CLIENT = genai.Client(api_key=GEMINI_API_KEY)
else:
    CLIENT = None
    print("Warning: GEMINI_API_KEY not found in .env. Falling back to rule-based assistant.")



# --- Curated Protein Knowledge Base ---

PROTEIN_KNOWLEDGE = {
    "TP53": {
        "common_name": "p53 / Tumor Protein p53",
        "function": "Acts as a tumor suppressor. It is a transcription factor that activates genes involved in cell cycle arrest, DNA repair, and apoptosis (programmed cell death). Often called the 'Guardian of the Genome'.",
        "disease": "Mutated in over 50% of human cancers, including breast, lung, colon, and ovarian cancers. Li-Fraumeni syndrome is caused by inherited TP53 mutations.",
        "drug_relevance": "A major drug target in oncology. Drugs like Nutlin-3 aim to reactivate p53 by blocking its interaction with MDM2.",
        "interactions": "Interacts with MDM2 (its primary negative regulator), BRCA1 (DNA repair), and BAX (apoptosis trigger).",
        "location": "Nucleus",
        "ensp_ids": ["ENSP00000269305", "ENSP00000398846"]
    },
    "BRCA1": {
        "common_name": "Breast Cancer Type 1 Susceptibility Protein",
        "function": "Involved in DNA double-strand break repair via homologous recombination. Critical for maintaining genomic stability.",
        "disease": "Mutations increase risk of breast cancer (~70% lifetime risk) and ovarian cancer (~40% lifetime risk). Also linked to prostate and pancreatic cancers.",
        "drug_relevance": "PARP inhibitors (Olaparib, Rucaparib) exploit BRCA1 deficiency through synthetic lethality — a Nobel Prize-winning concept.",
        "interactions": "Forms complexes with BARD1, RAD51, p53, and PALB2. The BRCA1-BARD1 complex is essential for its tumor suppressor function.",
        "location": "Nucleus",
        "ensp_ids": ["ENSP00000350283"]
    },
    "INS": {
        "common_name": "Insulin",
        "function": "A peptide hormone produced by the beta cells of the pancreatic islets of Langerhans. It regulates blood glucose levels by promoting cellular uptake of glucose.",
        "disease": "Deficiency or resistance leads to Diabetes Mellitus. Type 1 diabetes involves autoimmune destruction of insulin-producing cells. Type 2 involves insulin resistance.",
        "drug_relevance": "Recombinant insulin (Humalog, Lantus) is one of the most important biologic drugs in history, saving millions of lives since its discovery in 1921.",
        "interactions": "Binds to the Insulin Receptor (INSR), triggering a cascade involving IRS1, PI3K, and AKT signaling.",
        "location": "Secreted (extracellular)",
        "ensp_ids": ["ENSP00000381330"]
    },
    "EGFR": {
        "common_name": "Epidermal Growth Factor Receptor",
        "function": "A transmembrane receptor tyrosine kinase that triggers cell growth, division, and survival when activated by its ligands (EGF, TGF-alpha).",
        "disease": "Overexpression or mutations drive many cancers, especially Non-Small Cell Lung Cancer (NSCLC), colorectal cancer, and glioblastoma.",
        "drug_relevance": "Target of blockbuster drugs: Erlotinib (Tarceva), Gefitinib (Iressa) — small molecule inhibitors; Cetuximab (Erbitux) — monoclonal antibody. EGFR testing is required before prescribing these drugs.",
        "interactions": "Dimerizes with HER2 (ERBB2), activates RAS-MAPK and PI3K-AKT pathways.",
        "location": "Cell membrane",
        "ensp_ids": ["ENSP00000275493"]
    },
    "HER2": {
        "common_name": "Human Epidermal Growth Factor Receptor 2 (ERBB2)",
        "function": "A member of the ErbB receptor family. It has no known ligand but is the preferred dimerization partner for other ErbB receptors, amplifying growth signals.",
        "disease": "Amplified/overexpressed in ~20% of breast cancers (HER2-positive). Associated with aggressive tumor behavior and poor prognosis if untreated.",
        "drug_relevance": "Target of Trastuzumab (Herceptin) — a revolutionary monoclonal antibody that dramatically improved survival in HER2+ breast cancer. Also targeted by Pertuzumab and T-DM1.",
        "interactions": "Heterodimerizes with EGFR (HER1), HER3, and HER4. The HER2-HER3 dimer is the most potent oncogenic signal.",
        "location": "Cell membrane",
        "ensp_ids": ["ENSP00000269571"]
    },
    "ACE2": {
        "common_name": "Angiotensin-Converting Enzyme 2",
        "function": "A key enzyme in the renin-angiotensin system (RAS). Converts angiotensin II to angiotensin 1-7, which has vasodilatory and anti-inflammatory effects.",
        "disease": "Serves as the primary entry receptor for SARS-CoV-2, the virus that causes COVID-19. The viral spike protein binds ACE2 to enter human cells.",
        "drug_relevance": "A therapeutic target in COVID-19 research. Soluble ACE2 decoys and antibodies blocking the Spike-ACE2 interaction are active research areas.",
        "interactions": "The SARS-CoV-2 Spike protein RBD binds ACE2 with high affinity. Interacts with TMPRSS2 (serine protease) for viral entry.",
        "location": "Cell membrane (lungs, heart, kidneys, intestines)",
        "ensp_ids": ["ENSP00000389326"]
    },
    "APOE": {
        "common_name": "Apolipoprotein E",
        "function": "Mediates the transport and metabolism of cholesterol and other lipids. Essential for lipid homeostasis and neuronal repair.",
        "disease": "The APOE-ε4 allele is the strongest genetic risk factor for late-onset Alzheimer's disease (3-15x increased risk depending on copies). Also linked to cardiovascular disease.",
        "drug_relevance": "Intense research target for Alzheimer's prevention. APOE-targeted therapies, including antisense oligonucleotides and gene therapy, are in clinical trials.",
        "interactions": "Interacts with LDL receptors (LDLR family), amyloid-beta peptides, and tau protein in the brain.",
        "location": "Secreted (plasma, cerebrospinal fluid)",
        "ensp_ids": ["ENSP00000252486"]
    },
    "KRAS": {
        "common_name": "Kirsten Rat Sarcoma Viral Proto-Oncogene",
        "function": "A small GTPase that acts as a molecular switch in cell signaling. When activated by growth factor receptors, it turns on the MAPK/ERK pathway promoting cell growth.",
        "disease": "Mutated in ~25% of all human cancers. KRAS G12C mutation is particularly common in lung adenocarcinoma (~13%). Also mutated in pancreatic (>90%) and colorectal cancers.",
        "drug_relevance": "Was considered 'undruggable' for 40 years. Sotorasib (Lumakras) and Adagrasib (Krazati) — approved 2021/2022 — specifically target the KRAS G12C mutation, a breakthrough in oncology.",
        "interactions": "Activated by SOS1 (GEF), inactivated by NF1 (GAP). Signals through RAF-MEK-ERK and PI3K-AKT cascades.",
        "location": "Cell membrane (inner leaflet)",
        "ensp_ids": ["ENSP00000256078"]
    },
    "CFTR": {
        "common_name": "Cystic Fibrosis Transmembrane Conductance Regulator",
        "function": "An ion channel that transports chloride and bicarbonate ions across epithelial cell membranes. Regulates mucus consistency in the lungs and other organs.",
        "disease": "Mutations (most commonly F508del) cause Cystic Fibrosis — thick, sticky mucus in the lungs, pancreas, and other organs. Affects ~70,000 people worldwide.",
        "drug_relevance": "Trikafta (elexacaftor/tezacaftor/ivacaftor) — approved 2019 — is a revolutionary triple-combination therapy that corrects and potentiates defective CFTR. It treats ~90% of CF patients.",
        "interactions": "Interacts with PDZ domain proteins (NHERF1), cytoskeletal proteins, and other ion channels for proper membrane localization.",
        "location": "Cell membrane (epithelial cells)",
        "ensp_ids": []
    },
    "VEGFA": {
        "common_name": "Vascular Endothelial Growth Factor A",
        "function": "A signal protein that stimulates the formation of new blood vessels (angiogenesis). Critical during embryonic development and wound healing.",
        "disease": "Tumors hijack VEGF signaling to grow their own blood supply. Overexpression is linked to cancer progression, diabetic retinopathy, and age-related macular degeneration.",
        "drug_relevance": "Bevacizumab (Avastin) — a monoclonal antibody that blocks VEGF — is used in cancer and eye diseases. Ranibizumab (Lucentis) is used for retinal diseases. Combined, these represent a multi-billion dollar drug category.",
        "interactions": "Binds VEGFR1 (FLT1) and VEGFR2 (KDR). Neuropilin-1 serves as a co-receptor.",
        "location": "Secreted",
        "ensp_ids": ["ENSP00000478674"]
    },
    "TNF": {
        "common_name": "Tumor Necrosis Factor Alpha (TNF-α)",
        "function": "A pro-inflammatory cytokine involved in systemic inflammation. Plays key roles in immune defense, fever induction, and cell death signaling.",
        "disease": "Overproduction drives autoimmune diseases: Rheumatoid Arthritis, Crohn's Disease, Psoriasis, and Ankylosing Spondylitis.",
        "drug_relevance": "Anti-TNF biologics are among the best-selling drugs ever: Adalimumab (Humira — $200B+ lifetime sales), Infliximab (Remicade), Etanercept (Enbrel). They revolutionized autoimmune disease treatment.",
        "interactions": "Binds TNFR1 (ubiquitous) and TNFR2 (immune cells). Activates NF-κB and caspase-mediated apoptosis pathways.",
        "location": "Secreted / Cell membrane",
        "ensp_ids": ["ENSP00000449264"]
    },
}

# --- General Biology FAQ ---

GENERAL_FAQ = {
    "protein": "A **protein** is a large, complex molecule made up of chains of amino acids. Proteins do most of the work in cells and are required for the structure, function, and regulation of the body's tissues and organs. There are approximately 20,000 protein-coding genes in the human genome.",
    "ppi": "A **Protein-Protein Interaction (PPI)** occurs when two or more proteins physically bind to each other to perform a biological function. PPIs are fundamental to virtually every process in a living cell — from DNA replication to immune responses. Understanding PPIs helps us discover new drug targets and understand diseases.",
    "amino acid": "**Amino acids** are the building blocks of proteins. There are 20 standard amino acids, each with a unique side chain. They are linked together by peptide bonds to form polypeptide chains. The sequence of amino acids determines how a protein folds and what function it performs.",
    "enzyme": "An **enzyme** is a protein that acts as a biological catalyst — it speeds up chemical reactions in the body without being consumed. Examples include DNA polymerase (copies DNA), lactase (digests milk sugar), and trypsin (digests proteins in the stomach).",
    "antibody": "An **antibody** (immunoglobulin) is a Y-shaped protein produced by the immune system to neutralize pathogens like viruses and bacteria. Each antibody is specific to one antigen. Therapeutic antibodies like Trastuzumab (Herceptin) are now used as drugs.",
    "gene": "A **gene** is a segment of DNA that contains instructions for making a specific protein. Humans have about 20,000-25,000 genes. When a gene is 'expressed', its DNA is first transcribed into mRNA, which is then translated by ribosomes into a protein.",
    "mutation": "A **mutation** is a change in the DNA sequence of a gene. Mutations can be harmless (silent), beneficial, or harmful. Some mutations in key proteins (like p53 or BRCA1) can lead to diseases like cancer. Our TransGraph-PPI system can simulate how mutations affect protein interactions!",
    "drug target": "A **drug target** is a molecule in the body (usually a protein) that a drug binds to in order to produce its therapeutic effect. For example, ibuprofen targets the COX enzymes to reduce inflammation. Identifying the right drug target is the critical first step in developing new medicines.",
    "esm": "**ESM-2** (Evolutionary Scale Modeling) is a protein language model developed by Meta AI. It learns to understand protein sequences by training on millions of protein sequences from nature. Just like GPT understands text, ESM-2 understands the 'language' of proteins. Our TransGraph-PPI system uses ESM-2 to convert protein sequences into meaningful numerical representations (embeddings).",
    "gat": "A **Graph Attention Network (GAT)** is a type of neural network designed to work with graph-structured data. In our system, proteins are nodes and their interactions are edges. The GAT learns which neighboring proteins are most important for predicting new interactions, using an 'attention' mechanism to weigh connections differently.",
    "ensemble": "An **ensemble model** combines predictions from multiple individual models to make more accurate final predictions. In TransGraph-PPI, we combine the ESM-MLP (sequence-based) and GAT (graph-based) predictions using XGBoost — achieving better accuracy than either model alone.",
    "shap": "**SHAP (SHapley Additive exPlanations)** is a technique from game theory that explains how each feature contributes to a model's prediction. In our system, SHAP tells you how much the sequence model vs. the graph model influenced the final interaction prediction — making the AI transparent and trustworthy.",
    "string database": "The **STRING database** (Search Tool for the Retrieval of Interacting Genes/Proteins) is the world's largest database of known and predicted protein-protein interactions. It covers over 67 million proteins from 14,000+ organisms. Our TransGraph-PPI model is trained on high-confidence human interactions from STRING v12.0.",
    "uniprot": "**UniProt** (Universal Protein Resource) is the most comprehensive database of protein sequence and functional information. It contains over 250 million protein sequences. We use UniProt for protein sequence retrieval and ID mapping in our system.",
    "cell": "A **cell** is the basic unit of life. Human cells contain a nucleus (with DNA), mitochondria (energy production), ribosomes (protein synthesis), and many other organelles. Proteins interact with each other inside and outside cells to carry out all biological functions.",
    "dna": "**DNA (Deoxyribonucleic Acid)** is the molecule that carries genetic instructions for life. It consists of two strands forming a double helix, made up of four nucleotide bases: Adenine (A), Thymine (T), Guanine (G), and Cytosine (C). The sequence of these bases encodes the instructions for making proteins.",
    "rna": "**RNA (Ribonucleic Acid)** is a molecule similar to DNA but usually single-stranded. mRNA (messenger RNA) carries genetic instructions from DNA to ribosomes for protein synthesis. Other types include tRNA (transfer), rRNA (ribosomal), and miRNA (regulatory).",
    "cancer": "**Cancer** is a group of diseases characterized by uncontrolled cell growth caused by mutations in key proteins. Common mutated proteins include p53, KRAS, BRCA1, and EGFR. Understanding protein interactions helps identify new cancer drug targets. Our system can analyze these critical proteins!",
    "alzheimer": "**Alzheimer's disease** is a neurodegenerative disorder and the most common cause of dementia. It involves the accumulation of amyloid-beta plaques and tau tangles in the brain. The APOE-ε4 gene variant is the strongest genetic risk factor. Current research focuses on anti-amyloid antibodies like Lecanemab.",
    "covid": "**COVID-19** is caused by the SARS-CoV-2 virus. The virus enters human cells when its Spike protein binds to the ACE2 receptor on cell surfaces. Understanding this protein-protein interaction was crucial for developing vaccines and treatments. TMPRSS2 protease also plays a key role in viral entry.",
    "diabetes": "**Diabetes** is a metabolic disease involving high blood sugar. Type 1 is autoimmune (destroying insulin-producing cells). Type 2 involves insulin resistance. The insulin protein and its receptor (INSR) are central to the disease. Modern treatments include recombinant insulin analogs and GLP-1 receptor agonists (like Semaglutide/Ozempic).",
    "fold": "**Protein folding** is the process by which a polypeptide chain acquires its functional 3D structure. The sequence of amino acids determines the fold. Misfolded proteins can cause diseases like Alzheimer's (amyloid-beta), Parkinson's (alpha-synuclein), and prion diseases (PrP).",
    "transgraph": "**TransGraph-PPI** is our hybrid deep learning framework that predicts protein-protein interactions. It combines:\n\n🧬 **ESM-2** — A protein language model for sequence understanding\n🕸️ **Graph Attention Network (GAT)** — Learns interaction patterns from the protein network\n🎯 **XGBoost Ensemble** — Combines both models for maximum accuracy\n📊 **SHAP Explainability** — Explains why each prediction was made\n\nYou can try it on the **Predict Interaction** page!",
}

GREETINGS = [
    "Hello! 👋 I'm your Protein Discovery Assistant. Ask me about any protein, disease, or biology concept! For example, try asking 'What is p53?' or 'Tell me about BRCA1'.",
    "Hi there! 🧬 I'm here to help you explore the world of proteins. You can ask me things like 'What causes Alzheimer's?' or 'What is EGFR?'",
    "Welcome! 🔬 I can tell you about proteins, diseases, drug targets, and how our TransGraph-PPI system works. What would you like to know?",
]

SUGGESTIONS = [
    "What is p53?",
    "Tell me about BRCA1 and breast cancer",
    "How does insulin work?",
    "What is a protein-protein interaction?",
    "Explain how TransGraph-PPI works",
    "What is EGFR and why is it important in cancer?",
    "How does COVID-19 infect cells?",
    "What is SHAP explainability?",
    "Tell me about drug targets",
    "What causes Alzheimer's disease?",
    "What is the KRAS protein?",
    "How does the ensemble model work?",
]


class ProteinAssistant:
    """A knowledge-based AI assistant for protein biology questions."""

    def __init__(self, sequence_manager=None, target_manager=None):
        self.sequence_manager = sequence_manager
        self.target_manager = target_manager

    def get_greeting(self) -> dict:
        greeting_text = random.choice(GREETINGS)
        suggestions = random.sample(SUGGESTIONS, min(4, len(SUGGESTIONS)))
        return {
            "response": greeting_text,
            "suggestions": suggestions,
            "sources": []
        }

    def answer(self, question: str) -> dict:
        """
        Process a user question using Gemini AI or fallback to rule-based logic.
        """
        q = question.lower().strip()

        # 1. Check for greetings (keep it snappy)
        if q in ["hi", "hello", "hey", "help", "start"]:
            return self.get_greeting()

        # 2. Try Gemini AI if configured
        if CLIENT:
            try:
                # Prepare context from our curated knowledge
                context = "You are a Protein Biology Expert for the TransGraph-PPI project. "
                context += "Use the following curated knowledge if relevant, but answer the user's specific question naturally.\n\n"
                
                # Sample some knowledge for context (don't send everything to save tokens/time)
                relevant_keys = [k for k in PROTEIN_KNOWLEDGE.keys() if k.lower() in q]
                for k in relevant_keys:
                    context += f"Protein {k}: {PROTEIN_KNOWLEDGE[k]['function']}\n"
                
                prompt = f"{context}\n\nUser Question: {question}\n\nAssistant:"
                
                response = CLIENT.models.generate_content(
                    model='gemini-1.5-pro',
                    contents=prompt
                )
                
                if response and response.text:
                    return {
                        "response": response.text,
                        "suggestions": random.sample(SUGGESTIONS, 3),
                        "sources": ["Google Gemini AI", "TransGraph-PPI Knowledge Base"]
                    }
            except Exception as e:
                print(f"Gemini AI Error: {e}")
                # Fall through to rule-based

        # 3. Rule-based fallback (original logic)
        protein_result = self._search_protein(q)
        if protein_result:
            return protein_result

        faq_result = self._search_faq(q)
        if faq_result:
            return faq_result

        # 4. Check for disease-related queries
        disease_result = self._search_disease(q)
        if disease_result:
            return disease_result

        # 5. Check for "what can you do" type queries
        if any(kw in q for kw in ["what can you", "help me", "what do you", "your features", "capabilities"]):
            return {
                "response": "I can help you with many things! Here's what I know about:\n\n"
                           "🧬 **Specific Proteins** — Ask about p53, BRCA1, EGFR, KRAS, Insulin, ACE2, HER2, VEGF, TNF, APOE, CFTR and more\n\n"
                           "🦠 **Diseases** — Cancer, Alzheimer's, Diabetes, COVID-19, Cystic Fibrosis\n\n"
                           "💊 **Drug Targets** — How drugs work against specific proteins\n\n"
                           "🔬 **Biology Concepts** — Proteins, DNA, RNA, mutations, enzymes, antibodies\n\n"
                           "🤖 **TransGraph-PPI** — How our AI prediction system works (ESM-2, GAT, SHAP)\n\n"
                           "Just type your question naturally!",
                "suggestions": ["What is p53?", "How does TransGraph-PPI work?", "Tell me about cancer proteins", "What is a drug target?"],
                "sources": []
            }

        # 6. Fallback
        return {
            "response": f"I don't have specific information about '{question}' in my knowledge base yet. "
                       "However, I can tell you about many important proteins and biology concepts!\n\n"
                       "Try asking about specific proteins like **p53**, **BRCA1**, **EGFR**, **KRAS**, or **Insulin**, "
                       "or concepts like **protein-protein interactions**, **mutations**, or **drug targets**.",
            "suggestions": random.sample(SUGGESTIONS, 4),
            "sources": []
        }

    def _search_protein(self, query: str) -> Optional[dict]:
        """Search for a specific protein in the knowledge base."""
        # Normalize query: lowercase and strip punctuation
        q_clean = re.sub(r'[^\w\s-]', ' ', query.lower())
        
        for key, data in PROTEIN_KNOWLEDGE.items():
            # Build list of synonym terms
            synonyms = [key.lower()]
            
            # Split common name by common delimiters: /, (, ), comma
            cleaned_common = re.sub(r'[\(\)]', ' ', data["common_name"].lower())
            for part in re.split(r'[/,]', cleaned_common):
                part_stripped = part.strip()
                if part_stripped:
                    synonyms.append(part_stripped)
            
            # Add ENSP IDs
            synonyms.extend([e.lower() for e in data.get("ensp_ids", [])])
            
            # Check if any synonym matches as a whole phrase or word in the query
            matched = False
            for syn in synonyms:
                # Match it as whole words
                pattern = r'\b' + re.escape(syn) + r'\b'
                if re.search(pattern, q_clean):
                    matched = True
                    break
                    
            if matched:
                # Build comprehensive response
                response = f"## {data['common_name']} ({key})\n\n"
                response += f"**🔬 Function:** {data['function']}\n\n"
                response += f"**🏥 Disease Relevance:** {data['disease']}\n\n"
                response += f"**💊 Drug Relevance:** {data['drug_relevance']}\n\n"
                response += f"**🔗 Key Interactions:** {data['interactions']}\n\n"
                response += f"**📍 Location:** {data['location']}\n\n"

                if data.get("ensp_ids"):
                    ids = ", ".join(data["ensp_ids"])
                    response += f"**🆔 Ensembl IDs:** `{ids}`\n\n"
                    response += f"💡 *You can use these IDs on the **Predict Interaction** page to test interactions with other proteins!*"

                suggestions = [f"What diseases involve {key}?"]
                # Add some related suggestions
                other_proteins = [k for k in PROTEIN_KNOWLEDGE if k != key]
                suggestions.extend([f"Tell me about {p}" for p in random.sample(other_proteins, min(2, len(other_proteins)))])
                suggestions.append("How does TransGraph-PPI work?")

                sources = ["STRING Database v12.0", "UniProt", "NCBI Gene"]

                return {
                    "response": response,
                    "suggestions": suggestions,
                    "sources": sources
                }
        return None

    def _search_faq(self, query: str) -> Optional[dict]:
        """Search for matching FAQ topics."""
        best_match = None
        best_score = 0

        for topic, answer in GENERAL_FAQ.items():
            # Calculate a simple relevance score
            topic_words = topic.lower().split()
            score = sum(1 for w in topic_words if w in query)

            # Boost for exact topic match
            if topic.lower() in query:
                score += 3

            # Check for common question patterns
            if score > best_score:
                best_score = score
                best_match = (topic, answer)

        if best_match and best_score >= 1:
            topic, answer = best_match
            suggestions = random.sample([s for s in SUGGESTIONS], min(3, len(SUGGESTIONS)))
            return {
                "response": answer,
                "suggestions": suggestions,
                "sources": ["TransGraph-PPI Knowledge Base"]
            }
        return None

    def _search_disease(self, query: str) -> Optional[dict]:
        """Search for disease-related information across protein knowledge."""
        disease_keywords = {
            "cancer": ["TP53", "BRCA1", "EGFR", "HER2", "KRAS", "VEGFA"],
            "breast cancer": ["BRCA1", "HER2", "TP53"],
            "lung cancer": ["EGFR", "KRAS", "TP53"],
            "alzheimer": ["APOE"],
            "covid": ["ACE2"],
            "coronavirus": ["ACE2"],
            "sars": ["ACE2"],
            "diabetes": ["INS"],
            "cystic fibrosis": ["CFTR"],
            "autoimmune": ["TNF"],
            "rheumatoid": ["TNF"],
            "arthritis": ["TNF"],
            "crohn": ["TNF"],
        }

        for disease, proteins in disease_keywords.items():
            if disease in query:
                response = f"## Proteins involved in {disease.title()}\n\n"
                for pname in proteins:
                    pdata = PROTEIN_KNOWLEDGE.get(pname, {})
                    if pdata:
                        response += f"### {pdata['common_name']} ({pname})\n"
                        response += f"{pdata['disease']}\n\n"
                        response += f"**💊 Treatment:** {pdata['drug_relevance']}\n\n"

                suggestions = [f"Tell me about {p}" for p in proteins[:3]]
                suggestions.append("What is a drug target?")

                return {
                    "response": response,
                    "suggestions": suggestions,
                    "sources": ["STRING Database", "UniProt", "NCBI Gene", "ClinicalTrials.gov"]
                }
        return None
