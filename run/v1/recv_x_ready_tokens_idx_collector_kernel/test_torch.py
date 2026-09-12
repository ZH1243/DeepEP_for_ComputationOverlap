"""PyTorch wrapper checks with already-published input; run on a CUDA GPU."""

import unittest

import torch

try:
    from .collector import allocate_state, build, launch
except ImportError:
    from collector import allocate_state, build, launch


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class CollectorWrapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability()[0] < 9:
            raise unittest.SkipTest("requires Hopper or newer")
        build()

    def test_memberships_and_empty_ranges(self):
        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                host = torch.tensor([
                    [0, 3, -1], [2, -1, -1], [0, 0, -1], [3, -1, -1],
                    [0, 2, -1], [1, -1, -1], [3, 7, -1], [-1, -1, -1],
                ], dtype=dtype)
                topk = host.cuda()
                state = allocate_state(5, len(host))
                state.range_begin.copy_(torch.tensor([0, 0, 3, 3, 7], device="cuda"))
                state.range_end.copy_(torch.tensor([0, 3, 3, 7, 8], device="cuda"))
                state.ready_end.copy_(state.range_end)
                # Input is fully published before the collector starts here.
                # These ordinary stores are only safe because of this event;
                # an overlapping producer must use the CUDA release helpers.
                state.initialized.record()
                run = launch(topk, state)
                run.wait()
                for expert in range(8):
                    expected = torch.nonzero((host == expert).any(dim=1)).flatten().tolist()
                    count = state.ready_count[expert].item()
                    actual = state.indices[expert, :count].cpu().tolist()
                    self.assertEqual(sorted(actual), expected)
                with self.assertRaisesRegex(ValueError, "fresh state"):
                    launch(topk, state)

    def test_gather_bundles(self):
        for dtype in (torch.int32, torch.int64):
            for c, groups in ((3, 1), (17, 3), (512, 2)):
                with self.subTest(dtype=dtype, cluster_rows=c, groups=groups):
                    host = torch.tensor([[r % 7, r % 7, (r + 2) % 7]
                                         for r in range(1103)], dtype=dtype)
                    topk = host.cuda()
                    state = allocate_state(1, len(host), gather_cluster_rows=c,
                                           gather_num_n_groups=groups, gather_group_size=4)
                    state.range_begin.zero_()
                    state.range_end.fill_(len(host))
                    state.ready_end.copy_(state.range_end)
                    state.initialized.record()
                    launch(topk, state).wait()
                    counts = state.ready_count.cpu().tolist()
                    self.assertEqual(state.written_count.cpu().tolist(), counts)
                    q = state.gather_ready_rows.item()
                    self.assertEqual(q, sum((n + c - 1) // c for n in counts) * groups)
                    table = state.gather_table[:q].cpu().tolist()
                    gathered = [[] for _ in range(8)]
                    for r in range(0, q, groups):
                        expert, _, *indices = table[r]
                        valid = [i for i in indices if i >= 0]
                        self.assertGreater(len(valid), 0)
                        self.assertEqual(indices, valid + [-1] * (c - len(valid)))
                        for n in range(groups):
                            self.assertEqual(table[r + n], [expert, n * 4, *indices])
                        gathered[expert].extend(valid)
                    for expert in range(8):
                        expected = torch.nonzero((host == expert).any(dim=1)).flatten().tolist()
                        self.assertEqual(sorted(gathered[expert]), expected)
                        self.assertEqual(gathered[expert],
                                         state.indices[expert, :counts[expert]].cpu().tolist())

    def test_table_capacity_error(self):
        topk = torch.zeros((4, 1), device="cuda", dtype=torch.int64)
        state = allocate_state(1, 4, gather_cluster_rows=3, gather_num_n_groups=2)
        state.gather_table = state.gather_table[:1]  # Cannot commit a complete bundle.
        state.range_begin.zero_()
        state.range_end.fill_(4)
        state.ready_end.fill_(4)
        state.initialized.record()
        with self.assertRaisesRegex(RuntimeError, "gather table capacity exceeded"):
            launch(topk, state).wait()
        self.assertEqual(state.gather_ready_rows.item(), 0)
        self.assertEqual(state.written_count.cpu().tolist(), [0] * 8)

    def test_zero_rows(self):
        state = allocate_state(3, 0)
        state.range_begin.zero_()
        state.range_end.zero_()
        state.ready_end.zero_()
        state.initialized.record()
        launch(torch.empty((0, 8), device="cuda", dtype=torch.int64), state).wait()
        self.assertEqual(state.ready_count.cpu().tolist(), [0] * 8)
        self.assertEqual(state.written_count.cpu().tolist(), [0] * 8)
        self.assertEqual(state.gather_ready_rows.item(), 0)

    def test_capacity_error(self):
        topk = torch.zeros((2, 1), device="cuda", dtype=torch.int64)
        state = allocate_state(1, 2, capacity=1)
        state.range_begin.zero_()
        state.range_end.fill_(2)
        state.ready_end.fill_(2)
        state.initialized.record()
        run = launch(topk, state)
        with self.assertRaisesRegex(RuntimeError, "capacity exceeded"):
            run.wait()


if __name__ == "__main__":
    unittest.main()
