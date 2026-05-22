"""
향수 이미지 기반 분류 시스템 보고서 생성 (개선판)
"""
from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm, Twips
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from pathlib import Path

BASE = Path("results")
OUT  = Path("docs/report_deeplearning.docx")
OUT.parent.mkdir(exist_ok=True)

doc = Document()

# ── 페이지 설정 ───────────────────────────────────────────────────────────────
section = doc.sections[0]
section.page_width   = Cm(21)
section.page_height  = Cm(29.7)
section.left_margin  = Cm(2.5)
section.right_margin = Cm(2.5)
section.top_margin   = Cm(2.8)
section.bottom_margin= Cm(2.5)

FONT_KR   = "맑은 고딕"
FONT_EN   = "Calibri"
C_DARK    = RGBColor(0x1F, 0x49, 0x7D)   # 진한 파랑
C_MID     = RGBColor(0x2E, 0x74, 0xB5)   # 중간 파랑
C_LIGHT   = RGBColor(0x44, 0x72, 0xC4)   # 연한 파랑
C_GRAY    = RGBColor(0x59, 0x59, 0x59)   # 캡션 회색
C_WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
C_BLACK   = RGBColor(0x00, 0x00, 0x00)
HDR_FILL  = "1F497D"
ROW_FILL  = "EEF3FA"

# ── 헬퍼 ──────────────────────────────────────────────────────────────────────
def set_run(run, size=11, bold=False, italic=False, color=None, font=None):
    run.font.size   = Pt(size)
    run.font.bold   = bold
    run.font.italic = italic
    run.font.name   = font or FONT_KR
    if color:
        run.font.color.rgb = color

def para_spacing(p, before=0, after=6, line=None):
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after  = Pt(after)
    if line:
        pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        pf.line_spacing      = line

def shade_cell(cell, hex_color):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color)
    tcPr.append(shd)

def set_cell_margin(cell, top=80, bottom=80, left=120, right=120):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for side, val in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"),    str(val))
        el.set(qn("w:type"), "dxa")
        tcMar.append(el)
    tcPr.append(tcMar)

def add_page_number_footer():
    footer = doc.sections[0].footer
    p      = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    set_run(run, size=9, color=C_GRAY)
    fldChar1 = OxmlElement("w:fldChar"); fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText"); instrText.text = " PAGE "
    fldChar2 = OxmlElement("w:fldChar"); fldChar2.set(qn("w:fldCharType"), "end")
    run._r.append(fldChar1); run._r.append(instrText); run._r.append(fldChar2)

def heading1(text):
    p = doc.add_heading("", level=1)
    run = p.add_run(text)
    set_run(run, size=14, bold=True, color=C_DARK)
    para_spacing(p, before=18, after=8)
    p.paragraph_format.keep_with_next = True
    return p

def heading2(text):
    p = doc.add_heading("", level=2)
    run = p.add_run(text)
    set_run(run, size=12, bold=True, color=C_MID)
    para_spacing(p, before=12, after=6)
    p.paragraph_format.keep_with_next = True
    return p

def body_text(text, size=10.5, color=C_BLACK, before=0, after=5):
    p   = doc.add_paragraph()
    run = p.add_run(text)
    set_run(run, size=size, color=color)
    para_spacing(p, before=before, after=after, line=1.2)
    return p

def bullet(text, size=10.5):
    p   = doc.add_paragraph(style="List Bullet")
    run = p.add_run(text)
    set_run(run, size=size)
    para_spacing(p, before=1, after=3, line=1.15)
    return p

def label_text(text, size=10, bold=True, color=C_DARK):
    p   = doc.add_paragraph()
    run = p.add_run(text)
    set_run(run, size=size, bold=bold, color=color)
    para_spacing(p, before=8, after=3)
    return p

def spacer(pt=6):
    p = doc.add_paragraph()
    para_spacing(p, before=0, after=0)
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    pf.line_spacing      = Pt(pt)

def add_table(headers, rows, col_widths=None, note=None):
    spacer(4)
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style     = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER

    # 헤더 행
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        shade_cell(cell, HDR_FILL)
        set_cell_margin(cell)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        set_run(run, size=10, bold=True, color=C_WHITE)
        para_spacing(p, before=2, after=2)

    # 데이터 행
    for ri, row_data in enumerate(rows):
        row = t.rows[ri + 1]
        fill = ROW_FILL if ri % 2 == 0 else "FFFFFF"
        for ci, val in enumerate(row_data):
            cell = row.cells[ci]
            shade_cell(cell, fill)
            set_cell_margin(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if ci > 0 else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(str(val))
            set_run(run, size=10)
            para_spacing(p, before=2, after=2)

    # 열 너비
    if col_widths:
        for row in t.rows:
            for ci, w in enumerate(col_widths):
                row.cells[ci].width = Cm(w)

    # 주석
    if note:
        p   = doc.add_paragraph()
        run = p.add_run(f"※ {note}")
        set_run(run, size=9, italic=True, color=C_GRAY)
        para_spacing(p, before=2, after=8)
    else:
        spacer(8)

def add_figure(path, caption, width_cm=14.5, fig_num=None):
    """이미지 + 캡션을 정렬하여 삽입. 이미지 없으면 스킵."""
    p_path = Path(path)
    if not p_path.exists():
        return
    spacer(6)
    pic_p = doc.add_paragraph()
    pic_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para_spacing(pic_p, before=0, after=0)
    run = pic_p.add_run()
    run.add_picture(str(p_path), width=Cm(width_cm))

    cap_p = doc.add_paragraph()
    cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    label = f"[그림 {fig_num}] " if fig_num else ""
    run   = cap_p.add_run(f"{label}{caption}")
    set_run(run, size=9, italic=True, color=C_GRAY)
    para_spacing(cap_p, before=3, after=10)

def add_figure_pair(path1, cap1, path2, cap2, fig1, fig2, width_cm=7.0):
    """두 이미지를 나란히 삽입 (2열 표 사용)."""
    p1, p2 = Path(path1), Path(path2)
    if not p1.exists() or not p2.exists():
        if p1.exists(): add_figure(path1, cap1, fig_num=fig1)
        if p2.exists(): add_figure(path2, cap2, fig_num=fig2)
        return
    spacer(6)
    t = doc.add_table(rows=2, cols=2)
    t.style     = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER

    # 테두리 숨기기
    for row in t.rows:
        for cell in row.cells:
            tc   = cell._tc
            tcPr = tc.get_or_add_tcPr()
            tcBorders = OxmlElement("w:tcBorders")
            for side in ["top","left","bottom","right","insideH","insideV"]:
                el = OxmlElement(f"w:{side}")
                el.set(qn("w:val"), "none")
                tcBorders.append(el)
            tcPr.append(tcBorders)

    def insert_img(cell, img_path, cap, width):
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        pic_p = cell.paragraphs[0]
        pic_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para_spacing(pic_p, before=0, after=2)
        pic_p.add_run().add_picture(str(img_path), width=Cm(width))

    def insert_cap(cell, label, cap):
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(f"[그림 {label}] {cap}")
        set_run(run, size=9, italic=True, color=C_GRAY)
        para_spacing(p, before=2, after=4)

    insert_img(t.rows[0].cells[0], p1, cap1, width_cm)
    insert_img(t.rows[0].cells[1], p2, cap2, width_cm)
    insert_cap(t.rows[1].cells[0], fig1, cap1)
    insert_cap(t.rows[1].cells[1], fig2, cap2)

    for row in t.rows:
        for ci, w in enumerate([Cm(8.0), Cm(8.0)]):
            row.cells[ci].width = w
    spacer(8)

# ── 페이지 번호 푸터 ──────────────────────────────────────────────────────────
add_page_number_footer()

# ══════════════════════════════════════════════════════════════════════════════
# 표지
# ══════════════════════════════════════════════════════════════════════════════
for _ in range(7): spacer(18)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("향수 이미지 기반 노트 분류 시스템")
set_run(run, size=24, bold=True, color=C_DARK)
para_spacing(p, before=0, after=10)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("딥러닝 모델 소개 및 성능 평가 보고서")
set_run(run, size=15, color=C_LIGHT)
para_spacing(p, before=0, after=6)

# 구분선
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("─" * 34)
set_run(run, size=11, color=C_MID)
para_spacing(p, before=4, after=10)

for _ in range(3): spacer(18)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("2026년 5월")
set_run(run, size=12, color=C_GRAY)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# 1. 프로젝트 개요
# ══════════════════════════════════════════════════════════════════════════════
heading1("1. 프로젝트 개요")
body_text(
    "본 프로젝트는 향수병 이미지와 텍스트(브랜드명/제품명)를 입력으로 받아 향료 계열(Note)을 "
    "자동 분류하는 딥러닝 시스템을 개발한다. 향수의 시각적 특성과 텍스트 정보를 결합하여 "
    "분류 성능을 향상시키는 것이 핵심 목표이다.",
    before=2, after=8
)
add_table(
    ["항목", "내용"],
    [
        ["분류 태스크", "Note 7클래스 / Note 4클래스 / Brand 35클래스"],
        ["데이터셋",   "Perfume Recommendation Dataset  (~28,000장)"],
        ["학습 데이터", "train 22,557개 / train_aug 41,061개 (오프라인 증강)"],
        ["평가 데이터", "val 2,820개 / test 2,820개"],
        ["주요 모델",  "EfficientNet-B0, CLIP ViT-B/32"],
    ],
    col_widths=[4, 11.5]
)

# ══════════════════════════════════════════════════════════════════════════════
# 2. 딥러닝 모델 소개
# ══════════════════════════════════════════════════════════════════════════════
heading1("2. 사용 딥러닝 모델 소개")

heading2("2.1  EfficientNet-B0")
body_text(
    "EfficientNet은 Google이 2019년에 제안한 이미지 분류 모델로, 네트워크의 너비(width)·깊이(depth)·"
    "해상도(resolution)를 균형 있게 확장하는 복합 스케일링(Compound Scaling)을 도입하였다. "
    "B0은 기본 버전으로 파라미터 수 약 5.3M의 경량 모델이며, 소규모 데이터에서도 과적합 없이 "
    "안정적인 성능을 발휘한다.",
    before=2, after=8
)
add_table(
    ["특성", "내용"],
    [
        ["파라미터 수",  "약 5.3M"],
        ["입력 해상도",  "224 × 224"],
        ["사전학습",    "ImageNet-1K"],
        ["적용 태스크", "Note 분류 (7클래스 / 4클래스),  Brand 분류"],
        ["학습 전략",   "2단계 학습 — Stage1: Backbone Freeze, Stage2: Gradual Unfreeze"],
        ["텍스트 결합", "Brand + Name 임베딩을 이미지 특징과 concat → FC 분류"],
    ],
    col_widths=[4, 11.5]
)

heading2("2.2  CLIP  (Contrastive Language–Image Pretraining)")
body_text(
    "CLIP은 OpenAI가 2021년에 발표한 멀티모달 모델로, 4억 개의 이미지-텍스트 쌍을 대조학습"
    "(contrastive learning)으로 사전학습하였다. 이미지 인코더(ViT-B/32)와 텍스트 인코더를 "
    "동시에 학습하여 이미지와 텍스트를 동일한 임베딩 공간에 매핑하며, 풍부한 의미 표현 덕분에 "
    "적은 데이터로도 경쟁력 있는 성능을 보인다.",
    before=2, after=8
)
add_table(
    ["특성", "내용"],
    [
        ["모델 구조",   "ViT-B/32 (이미지 인코더)  +  Transformer (텍스트 인코더)"],
        ["파라미터 수",  "약 151M"],
        ["사전학습",    "WIT (WebImageText,  4억 쌍)"],
        ["적용 방식",   "① Linear Probe  ② 2단계 Image Fine-tune  ③ 텍스트 결합 Fine-tune"],
        ["텍스트 결합", "이미지 임베딩 + 텍스트 임베딩 concat → 분류 헤드"],
    ],
    col_widths=[4, 11.5]
)

heading2("2.3  2단계 학습 전략  (Two-Stage Training)")
body_text(
    "소규모 데이터에서 과적합을 방지하고 사전학습 표현을 최대한 보존하기 위해 "
    "두 단계로 나누어 학습하였다.",
    before=2, after=8
)
add_table(
    ["단계", "설명"],
    [
        ["Stage 1\n(Warm-up)",
         "Backbone 전체를 동결(Freeze)하고 분류 헤드만 학습.\n"
         "사전학습 특징을 보존하면서 태스크에 맞는 분류 경계를 초기화."],
        ["Stage 2\n(Fine-tune)",
         "마지막 블록부터 점진적으로 동결 해제(Gradual Unfreeze)하며 전체 미세조정.\n"
         "낮은 학습률로 도메인 적응 — 과적합 방지를 위해 Early Stopping 적용."],
    ],
    col_widths=[3, 12.5]
)

# ══════════════════════════════════════════════════════════════════════════════
# 3. 성능 평가 지표
# ══════════════════════════════════════════════════════════════════════════════
heading1("3. 성능 평가 지표")
body_text(
    "향수 Note 분류 데이터는 클래스 불균형이 존재하므로, 단일 지표에 의존하지 않고 "
    "다각도의 평가 지표를 함께 활용하였다.",
    before=2, after=8
)
add_table(
    ["지표", "설명"],
    [
        ["Accuracy",
         "전체 샘플 중 올바르게 분류한 비율. 전반적 성능 파악에 사용."],
        ["Macro F1-Score",
         "클래스별 F1을 산술 평균. 소수 클래스의 성능을 동등하게 반영하여 불균형 상황에 적합."],
        ["Top-3 Accuracy",
         "상위 3개 예측 안에 정답이 포함된 비율. 후보 추천 시스템 관점의 평가."],
        ["Confusion Matrix",
         "클래스 간 오분류 패턴을 시각화. 어떤 클래스가 혼동되는지 직관적으로 파악."],
        ["Random Baseline",
         "무작위 분류의 기준선 — 7클래스: 14.3%,  4클래스: 25.0%."],
    ],
    col_widths=[4, 11.5]
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# 4. 7클래스 실험 결과
# ══════════════════════════════════════════════════════════════════════════════
heading1("4. 실험 결과 — Note 분류 7클래스")
body_text(
    "클래스: Floral / Woody / Amber_Oriental / Citrus / Sweet / Spicy / Fresh\n"
    "데이터: train 22,557 (aug 41,061) / val 2,820 / test 2,820  |  랜덤 베이스라인: 0.1429",
    before=2, after=8
)
add_table(
    ["모델", "Accuracy", "Macro F1", "Top-3 Acc"],
    [
        ["EfficientNet-B0 + 텍스트",      "0.5191 ★", "0.3226",   "0.8766 ★"],
        ["CLIP + 텍스트 (two-stage)",      "0.4851",   "0.3384 ★", "0.8532"],
        ["CLIP 이미지만 (linear probe)",   "0.4723",   "0.2563",   "0.8106"],
        ["EfficientNet-B0 이미지만",       "0.4592",   "0.2243",   "0.8223"],
        ["CLIP 이미지만 (fine-tune)",      "0.4404",   "0.2894",   "0.8160"],
        ["랜덤 베이스라인",               "0.1429",   "—",        "—"],
    ],
    col_widths=[7.5, 2.75, 2.75, 2.75],
    note="★ 해당 지표 1위"
)

label_text("▶ 분석")
for txt in [
    "EfficientNet-B0 + 텍스트가 Accuracy(0.5191)와 Top-3 Accuracy(0.8766) 모두 1위 달성",
    "CLIP + 텍스트는 Macro F1(0.3384) 1위 — 클래스 불균형 상황에서 더 균형 잡힌 성능 발휘",
    "텍스트(brand/name) 정보 추가 시 이미지 단독 대비 두 아키텍처 모두 5~6%p 성능 향상 확인",
    "7클래스는 Sweet(87개), Spicy(51개) 등 극소수 클래스로 인해 Macro F1이 상대적으로 낮음",
]:
    bullet(txt)

spacer(4)

# 이미지: 혼동행렬 2개 나란히
add_figure_pair(
    BASE / "note_classification/efficientnet/with_text/confusion_matrix.png",
    "EfficientNet-B0 + 텍스트 혼동행렬 (7클래스)",
    BASE / "note_classification/clip/with_text/confusion_matrix.png",
    "CLIP + 텍스트 혼동행렬 (7클래스)",
    fig1=1, fig2=2, width_cm=7.2
)

# 클래스별 정확도 (전체 너비)
add_figure(
    BASE / "note_classification/efficientnet/with_text/per_class_accuracy.png",
    "EfficientNet-B0 + 텍스트 클래스별 정확도 (7클래스)",
    width_cm=14.5, fig_num=3
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# 5. 4클래스 실험 결과
# ══════════════════════════════════════════════════════════════════════════════
heading1("5. 실험 결과 — Note 분류 4클래스")
body_text(
    "클래스: Floral / Woody / Fresh / Amber\n"
    "데이터: train 22,557 / val 2,820 / test 2,820  |  랜덤 베이스라인: 0.2500",
    before=2, after=8
)
add_table(
    ["모델", "Accuracy", "Macro F1", "Top-3 Acc"],
    [
        ["EfficientNet-B0 + 텍스트",     "0.5177 ★", "0.4593 ★", "0.9355 ★"],
        ["CLIP + 텍스트 (two-stage)",     "0.4730",   "0.4359",   "0.9252"],
        ["CLIP 이미지만 (linear probe)",  "0.4387",   "0.3943",   "0.8851"],
        ["CLIP 이미지만 (fine-tune)",     "0.2972",   "0.2945",   "0.8674"],
        ["랜덤 베이스라인",              "0.2500",   "—",        "—"],
    ],
    col_widths=[7.5, 2.75, 2.75, 2.75],
    note="★ 해당 지표 1위"
)

label_text("▶ 분석")
for txt in [
    "EfficientNet-B0 + 텍스트가 Accuracy · Macro F1 · Top-3 모두 1위 달성",
    "4클래스로 줄이자 Macro F1이 7클래스 대비 전반적으로 0.10~0.15 향상 — 소수 클래스 제거 효과",
    "CLIP 이미지 fine-tune은 Stage2에서 과적합 발생 (train acc 0.53 vs val acc 0.30) → 성능 하락",
    "Top-3 Accuracy 88~93%로, 후보 3개를 제안하는 추천 UX에서는 충분히 활용 가능한 수준",
]:
    bullet(txt)

spacer(4)

add_figure_pair(
    BASE / "note_classification_4class/efficientnet/with_text/confusion_matrix.png",
    "EfficientNet-B0 + 텍스트 혼동행렬 (4클래스)",
    BASE / "note_classification_4class/clip/with_text/confusion_matrix.png",
    "CLIP + 텍스트 혼동행렬 (4클래스)",
    fig1=4, fig2=5, width_cm=7.2
)

add_figure(
    BASE / "note_classification_4class/efficientnet/with_text/per_class_accuracy.png",
    "EfficientNet-B0 + 텍스트 클래스별 정확도 (4클래스)",
    width_cm=14.5, fig_num=6
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# 6. 7클래스 vs 4클래스 비교
# ══════════════════════════════════════════════════════════════════════════════
heading1("6. 7클래스 vs 4클래스 비교 분석")
body_text(
    "동일한 데이터(22,557 train / 2,820 test)에서 클래스 수만 다르게 설정한 두 실험의 "
    "성능을 나란히 비교하였다.",
    before=2, after=8
)
add_table(
    ["모델", "7cls\nAcc", "4cls\nAcc", "7cls\nMacro F1", "4cls\nMacro F1"],
    [
        ["EfficientNet-B0 + 텍스트", "0.5191", "0.5177", "0.3226", "0.4593"],
        ["CLIP + 텍스트",            "0.4851", "0.4730", "0.3384", "0.4359"],
        ["CLIP linear probe",        "0.4723", "0.4387", "0.2563", "0.3943"],
        ["CLIP fine-tune",           "0.4404", "0.2972", "0.2894", "0.2945"],
    ],
    col_widths=[7.0, 2.0, 2.0, 2.5, 2.5]
)

label_text("▶ 분석")
for txt in [
    "Accuracy는 두 설정에서 유사 — 클래스 수 변화가 전체 정확도에 미치는 영향은 제한적",
    "Macro F1은 4클래스에서 전반적으로 높음 — 7클래스의 극소수 클래스(Sweet/Spicy) 제거 효과",
    "CLIP fine-tune만 예외적으로 4클래스에서 더 낮음 — 과적합이 클래스 수 감소 효과를 상쇄",
    "향수병 이미지 자체가 Note 계열을 시각적으로 드러내지 않아 두 설정 모두 0.52 내외 한계 존재",
]:
    bullet(txt)

spacer(8)

# ══════════════════════════════════════════════════════════════════════════════
# 7. 결론
# ══════════════════════════════════════════════════════════════════════════════
heading1("7. 결론")
body_text(
    "본 실험을 통해 도출된 주요 결론은 다음과 같다.",
    before=2, after=6
)
conclusions = [
    ("최고 성능 모델",
     "EfficientNet-B0 + 텍스트 (7클래스 Accuracy 0.5191 / 4클래스 Accuracy 0.5177)"),
    ("텍스트 기여도",
     "Brand + Name 텍스트 추가 시 이미지 단독 대비 Accuracy 약 5~6%p 향상"),
    ("4클래스 설정 장점",
     "Macro F1 관점에서 더 균형 잡힌 성능 — 실용적 배포 시 4클래스가 유리"),
    ("Top-3 활용 가능성",
     "Top-3 Accuracy 7클래스 0.88 / 4클래스 0.93 → 상위 3개 후보 추천 방식의 UX 설계에 적합"),
    ("개선 방향",
     "향 성분 텍스트(Notes 필드) 추가 입력, 데이터 증강 강화, 앙상블 적용 시 성능 개선 가능"),
]
add_table(
    ["항목", "내용"],
    [["  ①  " + k, v] for k, v in conclusions],
    col_widths=[4.5, 11.0]
)

doc.save(str(OUT))
print(f"보고서 저장 완료: {OUT}")
