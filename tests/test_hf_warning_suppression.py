import logging
import group_appeal_detector  # noqa: F401 — registers the filter as a side-effect


def test_unauthenticated_warning_is_suppressed(caplog):
    logger = logging.getLogger("huggingface_hub")
    with caplog.at_level(logging.WARNING, logger="huggingface_hub"):
        logger.warning("You are sending unauthenticated requests to the HF Hub.")
    assert not any("unauthenticated" in r.message.lower() for r in caplog.records)


def test_other_huggingface_hub_warnings_are_not_suppressed(caplog):
    logger = logging.getLogger("huggingface_hub")
    with caplog.at_level(logging.WARNING, logger="huggingface_hub"):
        logger.warning("Some other warning from huggingface_hub.")
    assert any("Some other warning" in r.message for r in caplog.records)
