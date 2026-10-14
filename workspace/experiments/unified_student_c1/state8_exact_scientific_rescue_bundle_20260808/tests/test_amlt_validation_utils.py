from scripts.amlt_validation_utils import target_matches


WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
SUBSCRIPTION = "b6dc87f3-c479-49c8-8cb5-7896da3ff895"


def test_target_matches_archived_feeds_encoding() -> None:
    target = {
        "cluster": "Feeds",
        "resource_group": "rg-cs-newsandfeeds-singularity",
        "service": "sing",
        "subscription_id": SUBSCRIPTION,
        "workspace_name": "CS-NewsAndFeeds-Singularity",
        "workspace_resource_group": "rg-cs-newsandfeeds-singularity",
        "workspace_subscription_id": SUBSCRIPTION,
    }
    assert target_matches(target, WORKSPACE)


def test_target_matches_rejects_archived_target_drift() -> None:
    target = {
        "cluster": "Feeds",
        "resource_group": "rg-cs-newsandfeeds-singularity",
        "service": "sing",
        "subscription_id": SUBSCRIPTION,
        "workspace_name": "different-workspace",
        "workspace_resource_group": "rg-cs-newsandfeeds-singularity",
        "workspace_subscription_id": SUBSCRIPTION,
    }
    assert not target_matches(target, WORKSPACE)

    target["workspace_name"] = "CS-NewsAndFeeds-Singularity"
    target["cluster"] = "different-target"
    assert not target_matches(target, WORKSPACE)
