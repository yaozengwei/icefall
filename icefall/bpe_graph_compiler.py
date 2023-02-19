# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                    Zengwei Yao)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pathlib import Path
from typing import List, Union

import k2
import sentencepiece as spm
import torch


class BpeCtcTrainingGraphCompiler(object):
    def __init__(
        self,
        lang_dir: Path,
        device: Union[str, torch.device] = "cpu",
        sos_token: str = "<sos/eos>",
        eos_token: str = "<sos/eos>",
    ) -> None:
        """
        Args:
          lang_dir:
            This directory is expected to contain the following files:

                - bpe.model
                - words.txt
          device:
            It indicates CPU or CUDA.
          sos_token:
            The word piece that represents sos.
          eos_token:
            The word piece that represents eos.
        """
        lang_dir = Path(lang_dir)
        model_file = lang_dir / "bpe.model"
        sp = spm.SentencePieceProcessor()
        sp.load(str(model_file))
        self.sp = sp
        self.word_table = k2.SymbolTable.from_file(lang_dir / "words.txt")
        self.device = device

        self.sos_id = self.sp.piece_to_id(sos_token)
        self.eos_id = self.sp.piece_to_id(eos_token)

        assert self.sos_id != self.sp.unk_id()
        assert self.eos_id != self.sp.unk_id()

    def texts_to_ids(self, texts: List[str]) -> List[List[int]]:
        """Convert a list of texts to a list-of-list of piece IDs.

        Args:
          texts:
            It is a list of strings. Each string consists of space(s)
            separated words. An example containing two strings is given below:

                ['HELLO ICEFALL', 'HELLO k2']
        Returns:
          Return a list-of-list of piece IDs.
        """
        return self.sp.encode(texts, out_type=int)

    def compile(
        self,
        piece_ids: List[List[int]],
        modified: bool = False,
    ) -> k2.Fsa:
        """Build a ctc graph from a list-of-list piece IDs.

        Args:
          piece_ids:
            It is a list-of-list integer IDs.
          modified:
           See :func:`k2.ctc_graph` for its meaning.
        Return:
          Return an FsaVec, which is the result of composing a
          CTC topology with linear FSAs constructed from the given
          piece IDs.
        """
        graph = k2.ctc_graph(piece_ids, modified=modified, device=self.device)
        return graph


def build_modified_ctc_topo(
    max_token: int,
    non_epsilon_first: bool = True,
    device: Union[str, torch.device] = "cpu",
) -> k2.Fsa:
    """Build a modified CTC topology.
    Modified from https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/k2/topologies.py  # noqa

    Args:
      max_token (int):
        The maximum token ID (inclusive).
        We assume that token IDs are contiguous (from 1 to max_token).
        0 represents blank.
      device:
        It indicates CPU or CUDA.
      non_epsilon_first:
        If True, in the generated modified CTC topology,
        the non-epsilon aux-label is on the first arc as normal.
        Otherwise, the non-epsilon aux-label is on the last arc.
    """
    num_states = max_token + 1
    blank_id = num_states  # replace blank id 0 with max_token + 1
    final_state = num_states

    arcs = f"0 0 {blank_id} 0 0.0\n"
    for i in range(1, num_states):
        if non_epsilon_first:
            arcs += f"0 {i} {i} {i} 0.0\n"
        else:
            arcs += f"0 {i} 0 0 0.0\n"
    arcs += f"0 {final_state} -1 -1 0.0\n"

    for i in range(1, num_states):
        if non_epsilon_first:
            arcs += f"{i} 0 0 0 0.0\n"
        else:
            arcs += f"{i} 0 {i} {i} 0.0\n"
        arcs += f"{i} {i} {i} 0 0.0\n"
    arcs += f"{final_state}"

    ctc_topo = k2.Fsa.from_str(arcs, num_aux_labels=1).to(device)
    ctc_topo = k2.arc_sort(ctc_topo)
    return ctc_topo


class ModifiedBpeCtcTrainingGraphCompiler(BpeCtcTrainingGraphCompiler):
    def __init__(
        self,
        lang_dir: Path,
        device: Union[str, torch.device] = "cpu",
        sos_token: str = "<sos/eos>",
        eos_token: str = "<sos/eos>",
        non_epsilon_first=True,
    ) -> None:
        """
        Args:
          lang_dir:
            This directory is expected to contain the following files:

                - bpe.model
                - words.txt
          device:
            It indicates CPU or CUDA.
          sos_token:
            The word piece that represents sos.
          eos_token:
            The word piece that represents eos.
          non_epsilon_first:
            If True, in the generated modified CTC topology,
            the non-epsilon aux-label is on the first arc as normal.
            Otherwise, the non-epsilon aux-label is on the last arc.
        """
        super().__init__(
            lang_dir=lang_dir, device=device, sos_token=sos_token, eos_token=eos_token
        )
        self.max_token_id = self.sp.get_piece_size() - 1

        # The blank label is set to max_token_id + 1
        self.ctc_topo = build_modified_ctc_topo(
            max_token=self.max_token_id,
            non_epsilon_first=non_epsilon_first,
            device=self.device,
        )

    def compile(
        self,
        piece_ids: List[List[int]],
        modified: bool = False,
    ) -> k2.Fsa:
        """Build a ctc graph from a list-of-list piece IDs.

        Args:
          piece_ids:
            It is a list-of-list integer IDs.
          modified:
           See :func:`k2.ctc_graph` for its meaning.

        Return:
          Return an FsaVec, which is the result of composing a
          CTC topology with linear FSAs constructed from the given
          piece IDs.
        """
        L = k2.linear_fst(labels=piece_ids, aux_labels=piece_ids).to(self.device)
        L = k2.add_epsilon_self_loops(L)
        ctc_topo_L = k2.compose(self.ctc_topo, L, treat_epsilons_specially=False)
        ctc_topo_L = k2.remove_epsilon(ctc_topo_L)
        # Now ctc_topo_L.labels is a RaggedTensor
        ctc_topo_L = k2.connect(ctc_topo_L)

        # Convert ctc_topo_L.labels from RaggedTensor to Tensor
        ctc_topo_L_invert = k2.invert(ctc_topo_L)
        ctc_topo_L.aux_labels = ctc_topo_L_invert.labels

        # Replace max_token_id + 1 with 0 for blank label
        labels = ctc_topo_L.labels.clone()
        labels[labels == self.max_token_id + 1] = 0
        ctc_topo_L.labels = labels

        return ctc_topo_L
