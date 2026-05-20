from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "tagger_prompt.py"


def _source_text() -> str:
    return SOURCE.read_text(encoding="utf-8")


def test_controls_and_image_upload_are_in_side_by_side_columns():
    source = _source_text()

    assert "eid_main_layout" in source
    assert "eid_controls_col" in source
    assert "eid_image_col" in source
    assert "with gr.Row(elem_id=eid_main_layout):" in source
    assert "with gr.Column(elem_id=eid_controls_col" in source
    assert "with gr.Column(elem_id=eid_image_col" in source


def test_tags_prompt_stays_below_two_column_layout():
    source = _source_text()

    main_layout_pos = source.index("with gr.Row(elem_id=eid_main_layout):")
    tags_pos = source.index('out_tags = gr.Textbox(label="Tags / Prompt"')
    send_pos = source.index('send_btn = gr.Button("Insert into Prompt")')

    assert main_layout_pos < tags_pos < send_pos


def test_negative_words_is_not_forced_to_one_line():
    source = _source_text()

    negative_words_start = source.index("negative_words = gr.Textbox(")
    negative_words_end = source.index("# IMPORTANT: unique per tab", negative_words_start)
    negative_words_block = source[negative_words_start:negative_words_end]

    assert "lines=1" not in negative_words_block
    assert "lines=2" in negative_words_block


def test_drop_zone_has_no_top_margin_in_right_column():
    source = _source_text()

    assert "#{eid_drop}{{position:relative;margin-top:0;" in source
