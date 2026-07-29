import csv
import re
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
V2=ROOT/'report/v2'

def pdf_pages(path):
    return len(re.findall(rb'/Type\s*/Page(?!s)',path.read_bytes()))

def test_v2_preserves_v1_and_canonical_values():
    assert (ROOT/'report/vggt_seminar_report.docx').is_file()
    assert (ROOT/'report/vggt_seminar_report.md').is_file()
    with (ROOT/'report/data/view_count_results.csv').open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    assert len(rows)==10
    assert min(float(r['peak_allocated_gib']) for r in rows)==5.260654926300049
    assert max(float(r['peak_allocated_gib']) for r in rows)==6.598845958709717

def test_v2_required_artifacts_exist():
    required=[V2/'vggt_seminar_report_v2.md',V2/'vggt_seminar_report_v2.docx',V2/'vggt_seminar_report_v2.pdf',V2/'supplementary/vggt_supplementary.pdf',V2/'diagrams/hero_visual.png',V2/'diagrams/study_at_a_glance.png',V2/'diagrams/vggt_architecture.png',V2/'diagrams/project_pipeline.png']
    assert all(p.is_file() and p.stat().st_size>1000 for p in required)
    assert len(list((V2/'figures').glob('*.png')))>=16

def test_animations_are_complete_and_deterministic_length():
    for scene in ('delivery_area','courtyard'):
        gif=V2/f'animations/{scene}_s10_rotation.gif'; mp4=V2/f'animations/{scene}_s10_rotation.mp4'
        assert gif.stat().st_size>100_000 and mp4.stat().st_size>100_000
        with Image.open(gif) as im: assert im.n_frames==80

def test_pdf_page_counts_and_selectable_text_are_reasonable():
    main=V2/'vggt_seminar_report_v2.pdf'; supp=V2/'supplementary/vggt_supplementary.pdf'
    assert 14<=pdf_pages(main)<=20
    assert 4<=pdf_pages(supp)<=12
    assert main.stat().st_size>500_000 and supp.stat().st_size>100_000

def test_v2_source_has_required_sections_citations_and_boundaries():
    text=(V2/'vggt_seminar_report_v2.md').read_text(encoding='utf-8')
    for heading in ('Abstract','Study at a Glance','Introduction','VGGT in Brief','Reproduction Setup','Experimental Methodology','Overlap-Aware Frame Selection','Experimental Results','Qualitative Results','Strengths and Failure Modes','Discussion','Limitations','Future Work','Conclusion','References'):
        assert f'## {heading}' in text
    assert 'DINOv2' in text and '[5]' in text
    assert 'confidence is model output - not calibrated accuracy' in text
    assert 'arbitrary unaligned units' in text
    assert 'no quantitative reconstruction-accuracy claim' in text.lower()
    assert 'Experimental Draft' not in text

def test_v2_builders_cannot_execute_inference():
    source='\n'.join((ROOT/p).read_text(encoding='utf-8').lower() for p in ('scripts/build_phase10_report_v2.py','scripts/build_v2_animations.py'))
    for forbidden in ('import torch','from torch','from vggt','import vggt','load_state_dict','from_pretrained','model('): assert forbidden not in source

def test_main_doc_has_captions_and_no_broken_local_markdown_links():
    text=(V2/'vggt_seminar_report_v2.md').read_text(encoding='utf-8')
    assert 'Figure 1.' in (ROOT/'scripts/build_phase10_report_v2.py').read_text(encoding='utf-8')
    assert 'Figure 17.' in (ROOT/'scripts/build_phase10_report_v2.py').read_text(encoding='utf-8')
    assert 'Table 4.' in (ROOT/'scripts/build_phase10_report_v2.py').read_text(encoding='utf-8')
    assert not re.findall(r'!\[[^]]*\]\(([^)]+)\)',text)
