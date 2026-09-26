"""
Script to generate publication-grade code snippet screenshots (Figs 5.1 - 5.6)
for the TransGraph-PPI project report.
Uses Pygments with PIL and CascadiaCode/Consolas for clean dark-mode IDE styling.
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pygments
from pygments.lexers import PythonLexer
from pygments.token import Token

OUTPUT_DIR = Path(r"e:\majorproject\assets\report_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Select font
FONT_PATHS = [
    r"C:\Windows\Fonts\CascadiaCode.ttf",
    r"C:\Windows\Fonts\consola.ttf",
]
CODE_FONT_PATH = None
for p in FONT_PATHS:
    if os.path.exists(p):
        CODE_FONT_PATH = p
        break

HEADER_FONT_PATH = r"C:\Windows\Fonts\arial.ttf" if os.path.exists(r"C:\Windows\Fonts\arial.ttf") else CODE_FONT_PATH

# One Dark / Modern VS Code theme colors
THEME = {
    "bg": (15, 23, 42),          # #0f172a slate-900 dark background
    "card_bg": (30, 41, 59),     # #1e293b slate-800 card
    "title_bar": (15, 23, 42),    # #0f172a
    "border": (51, 65, 85),      # #334155
    "text": (226, 232, 240),     # #e2e8f0 default text
    "line_num": (100, 116, 139), # #64748b muted line numbers
    "shadow": (0, 0, 0, 120),
    # Token colors
    Token.Keyword: (198, 120, 221),         # Purple
    Token.Keyword.Constant: (209, 154, 102),# Orange
    Token.Keyword.Namespace: (198, 120, 221),
    Token.Name.Class: (229, 192, 123),      # Gold
    Token.Name.Function: (97, 175, 239),    # Blue
    Token.Name.Builtin: (86, 182, 194),     # Cyan
    Token.Name.Decorator: (229, 192, 123),
    Token.String: (152, 195, 121),          # Green
    Token.Number: (209, 154, 102),          # Orange
    Token.Operator: (86, 182, 194),         # Cyan
    Token.Punctuation: (171, 178, 191),
    Token.Comment: (92, 99, 112),           # Slate Gray
    Token.Comment.Single: (92, 99, 112),
}

def get_token_color(tok):
    while tok:
        if tok in THEME:
            return THEME[tok]
        tok = tok.parent
    return THEME["text"]

def render_code_window(code_text: str, filename: str, figure_tag: str, start_line: int = 1) -> Image.Image:
    font_size = 20
    code_font = ImageFont.truetype(CODE_FONT_PATH, font_size)
    header_font = ImageFont.truetype(HEADER_FONT_PATH, 16)
    badge_font = ImageFont.truetype(HEADER_FONT_PATH, 14)

    lines = code_text.strip().split("\n")
    max_line_len = max(len(line) for line in lines)
    num_lines = len(lines)

    char_w = font_size * 0.62
    line_h = int(font_size * 1.55)

    line_num_w = int(len(str(start_line + num_lines)) * char_w + 35)
    padding_x = 40
    padding_y = 30
    title_bar_h = 50

    card_w = int(max(950, line_num_w + (max_line_len * char_w) + (padding_x * 2)))
    card_h = int(title_bar_h + (num_lines * line_h) + (padding_y * 2))

    outer_margin = 35
    img_w = card_w + outer_margin * 2
    img_h = card_h + outer_margin * 2

    # Canvas
    img = Image.new("RGBA", (img_w, img_h), (11, 15, 25, 255))
    draw = ImageDraw.Draw(img)

    # Card background with border
    card_rect = [outer_margin, outer_margin, outer_margin + card_w, outer_margin + card_h]
    draw.rounded_rectangle(card_rect, radius=12, fill=THEME["card_bg"], outline=THEME["border"], width=1)

    # Title bar
    title_bar_rect = [outer_margin, outer_margin, outer_margin + card_w, outer_margin + title_bar_h]
    draw.rounded_rectangle(title_bar_rect, radius=12, fill=THEME["title_bar"])
    # Square bottom of title bar
    draw.rectangle([outer_margin, outer_margin + title_bar_h - 12, outer_margin + card_w, outer_margin + title_bar_h], fill=THEME["title_bar"])
    draw.line([outer_margin, outer_margin + title_bar_h, outer_margin + card_w, outer_margin + title_bar_h], fill=THEME["border"], width=1)

    # Window controls (macOS style dots)
    dot_radius = 6
    dot_y = outer_margin + title_bar_h // 2
    dot_colors = [(255, 95, 86), (255, 189, 46), (39, 201, 63)] # red, yellow, green
    for i, color in enumerate(dot_colors):
        dot_x = outer_margin + 24 + i * 20
        draw.ellipse([dot_x - dot_radius, dot_y - dot_radius, dot_x + dot_radius, dot_y + dot_radius], fill=color)

    # Filename in header center
    title_text = f"{filename}"
    bbox = header_font.getbbox(title_text)
    tw = bbox[2] - bbox[0]
    draw.text((outer_margin + (card_w - tw) // 2, dot_y - 10), title_text, fill=(148, 163, 184), font=header_font)

    # Figure Badge in top right
    badge_bbox = badge_font.getbbox(figure_tag)
    bw = badge_bbox[2] - badge_bbox[0] + 16
    bh = 24
    bx = outer_margin + card_w - bw - 20
    by = dot_y - bh // 2
    draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=6, fill=(59, 130, 246, 60), outline=(96, 165, 250), width=1)
    draw.text((bx + 8, by + 3), figure_tag, fill=(191, 219, 254), font=badge_font)

    # Line number separator line
    draw.line([outer_margin + line_num_w + 10, outer_margin + title_bar_h + 10,
               outer_margin + line_num_w + 10, outer_margin + card_h - 15], fill=(51, 65, 85, 120), width=1)

    # Highlight and render code with Pygments
    lexer = PythonLexer()
    tokens = list(pygments.lex(code_text, lexer))

    # Token-based text positioning
    curr_line_idx = 0
    curr_x = outer_margin + line_num_w + 25
    curr_y = outer_margin + title_bar_h + padding_y

    # Draw first line number
    draw.text((outer_margin + line_num_w - 5, curr_y), f"{start_line}", fill=THEME["line_num"], font=code_font, anchor="ra")

    for tok_type, val in tokens:
        sub_tokens = val.split("\n")
        for i, sub in enumerate(sub_tokens):
            if i > 0:
                curr_line_idx += 1
                curr_y += line_h
                curr_x = outer_margin + line_num_w + 25
                if curr_line_idx < num_lines:
                    draw.text((outer_margin + line_num_w - 5, curr_y), f"{start_line + curr_line_idx}", fill=THEME["line_num"], font=code_font, anchor="ra")
            
            if sub:
                color = get_token_color(tok_type)
                draw.text((curr_x, curr_y), sub, fill=color, font=code_font)
                bbox = code_font.getbbox(sub)
                curr_x += (bbox[2] - bbox[0])

    return img


# -------------------------------------------------------------
# Snippet Definitions
# -------------------------------------------------------------

SNIPPETS = {
    "fig_5_1_dataset_preparation.png": {
        "tag": "Fig 5.1: Dataset Preparation",
        "file": "src/data/preprocess_data.py",
        "start_line": 46,
        "code": '''def load_string_interactions(filepath: str, min_score: int = 900) -> Tuple[pd.DataFrame, PairSet]:
    """
    Loads high-confidence STRING physical interactions (combined_score >= 900)
    and constructs a comprehensive non-zero exclusion set to prevent negative sample leakage.
    """
    df = pd.read_csv(filepath, sep=" ", usecols=["protein1", "protein2", "combined_score"])
    df["protein1"] = df["protein1"].astype(str).str.replace("9606.", "", regex=False)
    df["protein2"] = df["protein2"].astype(str).str.replace("9606.", "", regex=False)
    df = df[df["protein1"] != df["protein2"]]  # Remove homotypic self-loops

    a, b = _canonical(df["protein1"], df["protein2"])
    canon = pd.DataFrame({"protein1": a, "protein2": b, "combined_score": df["combined_score"].values})
    canon = canon.drop_duplicates(subset=["protein1", "protein2"]).reset_index(drop=True)

    positive_df = canon[canon["combined_score"] >= min_score][["protein1", "protein2"]]
    exclusion_pairs = PairSet(sorted(set(canon["protein1"]).union(canon["protein2"])),
                              canon["protein1"].values, canon["protein2"].values)
    return positive_df, exclusion_pairs

def generate_hard_negatives(positive_df, all_proteins, exclusion_pairs, ratio=0.5):
    """Generates topology-hard negative pairs sharing common neighbors but strictly non-interacting."""
    adj = defaultdict(set)
    for u, v in zip(positive_df["protein1"], positive_df["protein2"]):
        adj[u].add(v); adj[v].add(u)
    # Sample 2-hop non-edges with subcellular co-localization verification...'''
    },

    "fig_5_2_esm2_embedding_generation.png": {
        "tag": "Fig 5.2: ESM-2 Embedding Generation",
        "file": "src/data/feature_extraction.py",
        "start_line": 71,
        "code": '''class ESMFeatureExtractor:
    def __init__(self, model_name: str = "facebook/esm2_t30_150M_UR50D", device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def get_embeddings(self, sequences: Dict[str, str], batch_size: int = 4) -> Dict[str, torch.Tensor]:
        """Generates 640-dimensional masked mean-pooled representations from ESM-2 Transformer layers."""
        embeddings = {}
        ids = list(sequences.keys())
        seqs = [sequences[pid] for pid in ids]

        for i in tqdm(range(0, len(seqs), batch_size), desc="Extracting ESM-2 features"):
            batch_ids = ids[i:i + batch_size]
            batch_seqs = seqs[i:i + batch_size]

            # Tokenize sequence batch with explicit length cap (1024 tokens)
            inputs = self.tokenizer(batch_seqs, return_tensors="pt", padding=True,
                                    truncation=True, max_length=1024).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Compute attention-masked mean pooling across sequence length (excluding padding tokens)
                mask = inputs["attention_mask"].unsqueeze(-1).to(outputs.last_hidden_state.dtype)
                batch_embeddings = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)

            for pid, emb in zip(batch_ids, batch_embeddings):
                embeddings[pid] = emb.cpu().half()
        return embeddings'''
    },

    "fig_5_3_sequence_model.png": {
        "tag": "Fig 5.3: Sequence Model (Residual MLP)",
        "file": "src/models/sequence_model.py",
        "start_line": 7,
        "code": '''class SequencePPIModel(nn.Module):
    """High-Capacity Symmetric Residual MLP for Sequence-Based PPI Prediction (2560-d input)."""
    def __init__(self, input_dim: int = 640, hidden_dim: int = 1024, dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = input_dim * 4  # [u + v, u * v, |u - v|, max(u, v)]

        self.input_proj = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.res1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)
        )
        self.res2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2)
        )
        self.skip_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 1)
        )

    def forward(self, emb1, emb2):
        # Order-invariant symmetric operators for pairwise interaction
        f_sum, f_prod = emb1 + emb2, emb1 * emb2
        f_diff, f_max = torch.abs(emb1 - emb2), torch.max(emb1, emb2)
        x = torch.cat([f_sum, f_prod, f_diff, f_max], dim=1)

        x = F.gelu(self.res1(self.input_proj(x)) + self.input_proj(x))
        x = F.gelu(self.res2(x) + self.skip_proj(x))
        return self.head(x)'''
    },

    "fig_5_4_graph_model.png": {
        "tag": "Fig 5.4: Graph Model (GraphSAGE)",
        "file": "src/models/graph_model.py",
        "start_line": 6,
        "code": '''class SAGELinkPredictor(nn.Module):
    """2-Layer GraphSAGE Encoder + Bilinear Multi-Feature Decoder for PPI Link Prediction."""
    def __init__(self, in_channels: int = 643, hidden_channels: int = 256, dropout: float = 0.4):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_channels)
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = GNNBatchNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = GNNBatchNorm(hidden_channels)
        self.dropout = dropout

        # Decoder Head: [h_u, h_v, |h_u - h_v|, h_u * h_v, Bilinear(h_u, h_v)]
        self.bilinear = nn.Bilinear(hidden_channels, hidden_channels, 1)
        self.classifier = nn.Sequential(
            nn.Linear((hidden_channels * 4) + 1, 512),
            nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 1)
        )

    def encode(self, x, edge_index):
        x = F.dropout(torch.relu(self.bn1(self.conv1(self.input_norm(x), edge_index))), p=self.dropout, training=self.training)
        x = F.dropout(torch.relu(self.bn2(self.conv2(x, edge_index))), p=self.dropout, training=self.training)
        return x

    def decode(self, z, src, dst):
        h_u, h_v = z[src], z[dst]
        bilinear_out = self.bilinear(h_u, h_v)
        diff, hadamard = torch.abs(h_u - h_v), h_u * h_v
        pair_repr = torch.cat([h_u, h_v, diff, hadamard, bilinear_out], dim=1)
        return self.classifier(pair_repr)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index[0], edge_label_index[1])'''
    },

    "fig_5_5_xgboost_ensemble.png": {
        "tag": "Fig 5.5: XGBoost Stacking Ensemble",
        "file": "src/models/ensemble_model.py",
        "start_line": 14,
        "code": '''class PPIEnsemble:
    """Deep Stacking Ensemble using 7 Meta-Features and Out-of-Fold Trained XGBoost Classifier."""
    META_FEATURE_NAMES = ["p_seq", "p_graph", "conf_seq", "conf_graph", "diff", "max_conf", "consensus"]

    @staticmethod
    def _build_features(p_seq: np.ndarray, p_graph: np.ndarray) -> np.ndarray:
        conf_seq = np.abs(p_seq - 0.5)
        conf_graph = np.abs(p_graph - 0.5)
        disagreement = np.abs(p_seq - p_graph)
        max_conf = np.maximum(conf_seq, conf_graph)
        consensus = p_seq * p_graph  # Non-linear model agreement interaction
        return np.column_stack((p_seq, p_graph, conf_seq, conf_graph, disagreement, max_conf, consensus))

    def train_stacking(self, oof_seq_preds: np.ndarray, oof_graph_preds: np.ndarray, labels: np.ndarray):
        """Trains meta-learner on 5-fold out-of-fold base model predictions to prevent target leakage."""
        X_meta = self._build_features(oof_seq_preds, oof_graph_preds)
        self.meta_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            reg_alpha=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42
        )
        self.meta_model.fit(X_meta, labels)

    def predict(self, seq_probs: np.ndarray, raw_graph_probs: np.ndarray) -> np.ndarray:
        # Platt-scale raw graph probabilities, build 7-feature matrix, and query XGBoost
        calibrated_graph = self.calibrate_graph(raw_graph_probs)
        X_meta = self._build_features(np.asarray(seq_probs), calibrated_graph)
        return self.meta_model.predict_proba(X_meta)[:, 1]'''
    },

    "fig_5_6_shap_explainability.png": {
        "tag": "Fig 5.6: SHAP Explainability",
        "file": "src/analysis/explainability.py",
        "start_line": 8,
        "code": '''class PPIExplainer:
    """TreeSHAP Attribution Explainer for the 7 Meta-Features of the TransGraph-PPI Ensemble."""
    def __init__(self, meta_model_path: str):
        self.model = joblib.load(meta_model_path)
        self.explainer = shap.TreeExplainer(self.model)

    def explain_prediction(self, p_seq: float, p_graph: float, conf_seq: float,
                           conf_graph: float, diff: float, max_conf: float) -> np.ndarray:
        """Computes local Shapley values for an individual protein pair prediction."""
        consensus = p_seq * p_graph
        X = np.array([[p_seq, p_graph, conf_seq, conf_graph, diff, max_conf, consensus]])
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            return shap_values[1]  # Positive class attribution
        return shap_values

    def save_summary_plot(self, X: np.ndarray, feature_names=None, output_path="shap_summary.png"):
        """Generates beeswarm summary plot illustrating global impact of sequence vs. graph signals."""
        feature_names = feature_names or ["ESM-MLP", "GraphSAGE", "|ESM-0.5|",
                                          "|GraphSAGE-0.5|", "Disagreement", "Max Conf", "Consensus"]
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]

        plt.figure(figsize=(9, 5.5))
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.title("SHAP Meta-Feature Attribution Summary", fontsize=14, pad=12)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()'''
    }
}

if __name__ == "__main__":
    print("Generating code snippet screenshots...")
    for filename, config in SNIPPETS.items():
        img = render_code_window(
            code_text=config["code"],
            filename=config["file"],
            figure_tag=config["tag"],
            start_line=config["start_line"]
        )
        out_path = OUTPUT_DIR / filename
        img.save(out_path, dpi=(300, 300))
        print(f"  [Created] {out_path.name} ({img.width}x{img.height})")
    print("Done!")
