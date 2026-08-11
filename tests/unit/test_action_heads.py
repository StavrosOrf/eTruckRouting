import pytest


torch = pytest.importorskip("torch")

from algo.action_heads import (
    ACTION_HEAD_TYPES,
    CompleteGraphGCNActionHead,
    IndependentActionHead,
    SelfAttentionActionHead,
    build_action_head,
)


ENCODER_DIM = 6
HIDDEN_DIM = 8
ACTION_DIM = 5


def _head(head_type: str):
    torch.manual_seed(17)
    return build_action_head(
        head_type,
        ENCODER_DIM,
        HIDDEN_DIM,
        ACTION_DIM,
        num_layers=2,
        attention_heads=2,
        dropout=0.0,
    ).eval()


@pytest.mark.parametrize("head_type", ACTION_HEAD_TYPES)
def test_action_heads_are_permutation_equivariant(head_type):
    head = _head(head_type)
    embedding = torch.randn(2, ENCODER_DIM)
    actions = torch.randn(7, ACTION_DIM)
    ptr = torch.tensor([0, 4, 7])
    permutation_0 = torch.tensor([2, 0, 3, 1])
    permutation_1 = torch.tensor([1, 2, 0])
    permuted_actions = torch.cat(
        (actions[:4][permutation_0], actions[4:][permutation_1]), dim=0
    )

    with torch.no_grad():
        original = head(embedding, actions, ptr).logits
        permuted = head(embedding, permuted_actions, ptr).logits

    expected = torch.cat(
        (original[:4][permutation_0], original[4:][permutation_1]), dim=0
    )
    torch.testing.assert_close(permuted, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("head_type", ACTION_HEAD_TYPES)
def test_action_heads_do_not_leak_across_batch_segments(head_type):
    head = _head(head_type)
    embedding = torch.randn(2, ENCODER_DIM)
    actions = torch.randn(7, ACTION_DIM)
    ptr = torch.tensor([0, 3, 7])

    changed_embedding = embedding.clone()
    changed_embedding[1] = 1_000.0 * torch.randn(ENCODER_DIM)
    changed_actions = actions.clone()
    changed_actions[3:] = 1_000.0 * torch.randn(4, ACTION_DIM)

    with torch.no_grad():
        original = head(embedding, actions, ptr).logits
        changed = head(changed_embedding, changed_actions, ptr).logits

    torch.testing.assert_close(changed[:3], original[:3], rtol=0.0, atol=0.0)


def test_independent_head_has_no_within_set_action_leakage():
    head = _head("independent")
    embedding = torch.randn(1, ENCODER_DIM)
    actions = torch.randn(4, ACTION_DIM)
    ptr = torch.tensor([0, 4])
    changed_actions = actions.clone()
    changed_actions[1] += 100.0

    with torch.no_grad():
        original = head(embedding, actions, ptr).logits
        changed = head(embedding, changed_actions, ptr).logits

    torch.testing.assert_close(
        changed[torch.tensor([0, 2, 3])],
        original[torch.tensor([0, 2, 3])],
        rtol=0.0,
        atol=0.0,
    )


def test_complete_graph_contains_every_ordered_pair_without_cross_batch_edges():
    ptr = torch.tensor([0, 3, 5])
    edge_index = CompleteGraphGCNActionHead.build_complete_edge_index(ptr)
    actual = set(zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True))
    expected = {
        (source, target)
        for start, end in ((0, 3), (3, 5))
        for source in range(start, end)
        for target in range(start, end)
        if source != target
    }

    assert actual == expected
    assert edge_index.shape == (2, 8)


@pytest.mark.parametrize("head_type", ACTION_HEAD_TYPES)
def test_action_heads_support_empty_and_singleton_segments(head_type):
    head = _head(head_type)
    embedding = torch.randn(4, ENCODER_DIM)
    actions = torch.randn(3, ACTION_DIM)
    ptr = torch.tensor([0, 0, 1, 3, 3])

    output = head(embedding, actions, ptr)

    assert output.logits.shape == (3,)
    assert torch.equal(output.ptr, ptr)
    assert torch.isfinite(output.logits).all()


@pytest.mark.parametrize("head_type", ACTION_HEAD_TYPES)
def test_action_heads_support_an_all_empty_batch(head_type):
    head = _head(head_type)
    output = head(
        torch.randn(2, ENCODER_DIM),
        torch.empty((0, ACTION_DIM)),
        torch.tensor([0, 0, 0]),
    )

    assert output.logits.shape == (0,)


@pytest.mark.parametrize("head_type", ACTION_HEAD_TYPES)
def test_action_head_gradients_are_finite(head_type):
    head = _head(head_type).train()
    embedding = torch.randn(2, ENCODER_DIM, requires_grad=True)
    actions = torch.randn(5, ACTION_DIM, requires_grad=True)
    ptr = torch.tensor([0, 2, 5])

    loss = head(embedding, actions, ptr).logits.square().mean()
    loss.backward()

    assert embedding.grad is not None and torch.isfinite(embedding.grad).all()
    assert actions.grad is not None and torch.isfinite(actions.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in head.parameters()
    )


@pytest.mark.parametrize(
    ("head_type", "expected_type"),
    (
        ("independent", IndependentActionHead),
        ("complete-gcn", CompleteGraphGCNActionHead),
        ("self-attention", SelfAttentionActionHead),
    ),
)
def test_action_head_factory_normalizes_names(head_type, expected_type):
    assert isinstance(
        build_action_head(
            head_type,
            ENCODER_DIM,
            HIDDEN_DIM,
            ACTION_DIM,
            attention_heads=2,
        ),
        expected_type,
    )


def test_action_head_factory_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown action head"):
        build_action_head("chain", ENCODER_DIM, HIDDEN_DIM, ACTION_DIM)


@pytest.mark.parametrize(
    ("embedding", "actions", "ptr", "error", "match"),
    (
        (
            torch.randn(2, ENCODER_DIM),
            torch.randn(3, ACTION_DIM),
            torch.tensor([1, 2, 3]),
            ValueError,
            "start at zero",
        ),
        (
            torch.randn(2, ENCODER_DIM),
            torch.randn(3, ACTION_DIM),
            torch.tensor([0, 3, 2]),
            ValueError,
            "end at the number",
        ),
        (
            torch.randn(2, ENCODER_DIM),
            torch.randn(3, ACTION_DIM),
            torch.tensor([0.0, 1.0, 3.0]),
            TypeError,
            "integer dtype",
        ),
        (
            torch.randn(2, ENCODER_DIM),
            torch.randn(3, ACTION_DIM),
            torch.tensor([0, 3]),
            ValueError,
            r"batch_size \+ 1",
        ),
    ),
)
def test_action_head_rejects_malformed_pointers(embedding, actions, ptr, error, match):
    with pytest.raises(error, match=match):
        _head("independent")(embedding, actions, ptr)


def test_action_head_rejects_nonfinite_inputs():
    actions = torch.randn(2, ACTION_DIM)
    actions[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        _head("complete_gcn")(
            torch.randn(1, ENCODER_DIM), actions, torch.tensor([0, 2])
        )


def test_attention_configuration_validation():
    with pytest.raises(ValueError, match="divisible"):
        SelfAttentionActionHead(ENCODER_DIM, 7, ACTION_DIM, num_heads=2)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        SelfAttentionActionHead(
            ENCODER_DIM, HIDDEN_DIM, ACTION_DIM, num_heads=2, dropout=1.0
        )
