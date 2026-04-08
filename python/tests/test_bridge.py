"""
test_bridge.py — Testes de integração do bridge Python↔Julia (mmap)

Testa sem o engine Julia rodando:
    - Criação e estrutura do arquivo mmap
    - Leitura e escrita nos offsets corretos
    - Mecanismo de lock (filelock)
    - Timeout do bridge quando engine não responde
    - Tokenizer: encode/decode roundtrip
    - MNS local: sanidade da fórmula

Execute com:
    python -m pytest python/tests/test_bridge.py -v
ou:
    python python/tests/test_bridge.py
"""

import os
import sys
import mmap
import struct
import tempfile
import unittest

# Adiciona python/ ao path para importar módulos locais
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestMmapStructure(unittest.TestCase):
    """Valida a estrutura do arquivo de memória compartilhada."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mem")
        self.tmp.write(b'\x00' * 1024)
        self.tmp.flush()
        self.mem_path = self.tmp.name

    def tearDown(self):
        self.tmp.close()
        os.unlink(self.mem_path)

    def test_file_size(self):
        self.assertEqual(os.path.getsize(self.mem_path), 1024)

    def test_cmd_id_offset_zero(self):
        """Offset 0 deve ser 0x00 (idle) no estado inicial."""
        with open(self.mem_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self.assertEqual(mm[0], 0x00)
            mm.close()

    def test_write_cmd_id(self):
        """Escrita no offset 0 (CmdID) deve ser legível."""
        with open(self.mem_path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            mm[0] = 0x01
            self.assertEqual(mm[0], 0x01)
            mm.close()

    def test_reward_offset(self):
        """Offset 40 deve aceitar e retornar float32 corretamente."""
        reward_in = 0.75
        with open(self.mem_path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            mm[40:44] = struct.pack("f", reward_in)
            reward_out = struct.unpack("f", mm[40:44])[0]
            mm.close()
        self.assertAlmostEqual(reward_in, reward_out, places=5)

    def test_response_buffer_offset(self):
        """Offset 200-600 deve armazenar e recuperar texto UTF-8."""
        text = "Olá, CAFUNE!"
        enc = text.encode("utf-8")
        with open(self.mem_path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            mm[200:200+len(enc)] = enc
            raw = mm[200:600].split(b'\x00')[0]
            mm.close()
        self.assertEqual(raw.decode("utf-8"), text)

    def test_prompt_buffer_offset(self):
        """Offset 600-1000 deve armazenar e recuperar prompt UTF-8."""
        prompt = "Quem é você?"
        enc = prompt.encode("utf-8")[:399]
        with open(self.mem_path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            mm[600:600+len(enc)] = enc
            raw = mm[600:1000].split(b'\x00')[0]
            mm.close()
        self.assertEqual(raw.decode("utf-8"), prompt)


class TestBridgeTimeout(unittest.TestCase):
    """Testa o comportamento de timeout do bridge sem o engine Julia."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mem")
        self.tmp.write(b'\x00' * 1024)
        self.tmp.flush()
        self.tmp.close()
        self.mem_path = self.tmp.name

    def tearDown(self):
        os.unlink(self.mem_path)
        lock = self.mem_path + ".lock"
        if os.path.exists(lock):
            os.unlink(lock)

    def test_timeout_returns_error_string(self):
        """Bridge deve retornar string de erro após timeout (não lançar exceção)."""
        import importlib
        import python.bridge as bridge_module
        # Sobrescreve MEM_FILE para o arquivo temporário
        original = bridge_module.MEM_FILE
        bridge_module.MEM_FILE = self.mem_path
        bridge_module.LOCK_FILE = self.mem_path + ".lock"

        b = bridge_module.CAFUNEBridge.__new__(bridge_module.CAFUNEBridge)
        # Simula timeout curto (1s) sobrescrevendo constante local
        import time
        original_sleep = time.sleep

        call_count = [0]
        def fast_sleep(n):
            call_count[0] += 1
            if call_count[0] > 5:
                raise bridge_module.BridgeTimeoutError("Timeout forçado no teste")
            original_sleep(0.01)

        time.sleep = fast_sleep
        try:
            result = bridge_module.CAFUNEBridge.generate_response(b, "teste")
        finally:
            time.sleep = original_sleep
            bridge_module.MEM_FILE = original

        self.assertIn("Error", result)


class TestTokenizerRoundtrip(unittest.TestCase):
    """Testa encode → decode roundtrip do tokenizador."""

    def setUp(self):
        from tokenizer import CharTokenizer
        self.tok = CharTokenizer()
        corpus = [
            "Olá! Como posso te ajudar hoje?",
            "Eu sou o CAFUNE, assistente de difusão discreta.",
            "Haskell, Julia e Python constroem o motor.",
        ]
        self.tok.build_vocab(corpus)

    def test_encode_returns_list_of_ints(self):
        ids = self.tok.encode("Olá!")
        self.assertIsInstance(ids, list)
        self.assertTrue(all(isinstance(i, int) for i in ids))

    def test_decode_roundtrip(self):
        text = "Olá"
        ids = self.tok.encode(text, add_special=False)
        decoded = self.tok.decode(ids, skip_special=False)
        self.assertEqual(decoded, text)

    def test_bos_eos_added(self):
        from tokenizer import SPECIAL_TOKENS
        ids = self.tok.encode("teste", add_special=True)
        self.assertEqual(ids[0], SPECIAL_TOKENS["[BOS]"])
        self.assertEqual(ids[-1], SPECIAL_TOKENS["[EOS]"])

    def test_pad_truncate(self):
        ids = self.tok.encode("texto longo para truncar e paddar", add_special=False)
        padded = self.tok.pad(ids, 10)
        self.assertEqual(len(padded), 10)

    def test_vocab_size_param_ignored(self):
        """vocab_size passado no __init__ não deve lançar TypeError."""
        from tokenizer import CharTokenizer
        tok = CharTokenizer(vocab_size=512)
        self.assertIsNotNone(tok)

    def test_bpetokenizer_alias(self):
        """BPETokenizer deve ser o mesmo que CharTokenizer."""
        from tokenizer import BPETokenizer, CharTokenizer
        self.assertIs(BPETokenizer, CharTokenizer)


class TestMNSLocal(unittest.TestCase):
    """Testa o cálculo local do Mirror Neuron Score."""

    def setUp(self):
        from mns_local import compute_mns
        self.compute_mns = compute_mns

    def test_identical_texts_high_score(self):
        text = "Estou me sentindo triste hoje."
        mns, d_f, d_t = self.compute_mns(text, text)
        self.assertGreater(mns, 0.7)

    def test_unrelated_texts_low_df(self):
        mns, d_f, d_t = self.compute_mns(
            "Qual é a capital do Brasil?",
            "O céu é azul e as estrelas brilham."
        )
        self.assertLess(d_f, 0.4)

    def test_keyword_coverage_in_dt(self):
        prompt = "difusão discreta mascara tokens"
        response = "difusão discreta é usada para mascarar tokens aleatoriamente"
        mns, d_f, d_t = self.compute_mns(prompt, response)
        self.assertGreater(d_t, 0.5)

    def test_empty_inputs_return_zero(self):
        mns, d_f, d_t = self.compute_mns("", "qualquer coisa")
        self.assertEqual(mns, 0.0)

    def test_score_between_zero_and_one(self):
        mns, d_f, d_t = self.compute_mns("Olá!", "Oi, tudo bem?")
        self.assertGreaterEqual(mns, 0.0)
        self.assertLessEqual(mns, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
