"""
Acceptors for JSON parsing or constraning LLM generation to JSON outputs.
"""

import json

from .acceptor import (
    TokenAcceptor,
    CharAcceptor,
    StateMachineAcceptor,
    SequenceAcceptor,
    TextAcceptor,
)
from .util.tokentrie import TokenTrie


class WhitespaceTokenTrie(TokenTrie):
    """
    Create a smaller trie by collapsing all whitespace to a single one while
    keeping the token ids. Since all whitespace is equivalent in JSON, tokens
    that only differ in the leading amount of whitesspace are equivalent from
    an semantic point of view.

    For example, the tokens "\n" and "  " are both mapped to the same node as
    token " ", which now contains both their ids in its set of ids.

    This allows us to reduce the number of equivalent branches we explore when
    finding valid tokens. Note that this doesn't limit the possible output of
    an LLM, since the token ids are kept in the trie.
    """

    @classmethod
    def from_trie(cls, trie, whitespace_charset):
        """
        Create a WhitespaceTokenTrie given a full vocabulary trie.
        """
        if isinstance(trie, WhitespaceTokenTrie):
            return trie
        collapsed_trie = WhitespaceTokenTrie()
        for char, child in trie.children.items():
            # The trie doesn't need to contain tokens that don't start with whitespace,
            # since they won't be selected by the WhitespaceAcceptor.
            if char in whitespace_charset:
                for token, ids in child.dfs():
                    non_whitespace_index = next(
                        (
                            i
                            for i, char in enumerate(token)
                            if char not in whitespace_charset
                        ),
                        len(token),
                    )
                    collapsed_token = " " + token[non_whitespace_index:]
                    collapsed_trie.insert_ids(collapsed_token, ids)
        return collapsed_trie


class WhitespaceAcceptor(TokenAcceptor):
    """
    Optional whitespace
    """

    WHITESPACE = " \n\r\t"
    MAX_WHITESPACE = 40

    _cached_tries = {}

    @classmethod
    def prepare_trie(cls, trie: TokenTrie):
        """
        Build a collapsed trie that reduces the search space for valid tokens.
        """
        trie_id = id(trie)
        if trie_id in cls._cached_tries:
            return cls._cached_tries[trie_id]
        collapsed_trie = WhitespaceTokenTrie.from_trie(
            trie, WhitespaceAcceptor.WHITESPACE
        )
        cls._cached_tries[trie_id] = collapsed_trie
        return collapsed_trie

    class Cursor(TokenAcceptor.Cursor):
        """
        Cursor for WhitespaceAcceptor
        """

        def __init__(self, acceptor, text=""):
            self.text = text

        def select(self, candidate_chars):
            return WhitespaceAcceptor.WHITESPACE

        def prune(self, trie):
            """
            Use a custom matching trie to collapse all equivalent whitespace
            into one, saving time when selecting valid tokens.
            """
            return super().prune(WhitespaceAcceptor.prepare_trie(trie))

        def advance(self, char):
            # Sometimes, LLMs try to run away with spaces when they don't know how to continue.
            if len(self.text) > WhitespaceAcceptor.MAX_WHITESPACE:
                return []
            return [WhitespaceAcceptor.Cursor(None, self.text + char)]

        def in_accepted_state(self):
            return True

        def get_value(self):
            return self.text


class BooleanAcceptor(StateMachineAcceptor):
    """
    Accepts a JSON boolean value: true, false
    """

    def __init__(self):
        super().__init__([[(TextAcceptor("true"), "$"), (TextAcceptor("false"), "$")]])

    class Cursor(StateMachineAcceptor.Cursor):
        """
        Cursor for BooleanAcceptor
        """

        def __init__(self, acceptor):
            super().__init__(acceptor)
            self.value = None

        def can_complete_transition(self, transition_value, target_state, is_end_state):
            if is_end_state:
                if transition_value == "true":
                    self.value = True
                else:
                    assert transition_value == "false"
                    self.value = False
            return True

        def get_value(self):
            return self.value


class NullAcceptor(TextAcceptor):
    """
    Accepts the JSON null value
    """

    def __init__(self):
        super().__init__("null")


DigitAcceptor = CharAcceptor("0123456789")
HexDigitAcceptor = CharAcceptor("0123456789ABCDEFabcdef")


class StringCharTokenTrie(TokenTrie):
    """
    Create a smaller trie by collapsing all unescaped valid string characters
    to a single one while keeping the token ids. This is useful to reduce
    combinatorial explosion in string acceptance when all strings of equal
    length are equally acceptable.
    """

    @classmethod
    def from_trie(cls, trie):
        """
        Create a StringCharTokenTrie given a full trie.
        """
        if isinstance(trie, StringCharTokenTrie):
            return trie

        def _string_char_acceptor_collapse_fn(char):
            if char in ['"', "\\"]:
                return True
            if char in StringCharAcceptor.INVALID_CHARS:
                return None
            return "."

        # pylint: disable-next=protected-access
        return trie._map(_string_char_acceptor_collapse_fn, StringCharTokenTrie())


class StringCharAcceptor(TokenAcceptor):
    """
    Accepts a valid JSON unescaped string character
    """

    INVALID_CHARS = set(chr(c) for c in range(0, 0x20)) | set(['"', "\\"])
    _cached_tries = {}

    @classmethod
    def prepare_trie(cls, trie: TokenTrie):
        """
        Build a collapsed trie that reduces the search space for valid tokens.
        Note that while there is only one main vocabulary trie, we may need to
        several collapsed tries because sometimes string matching will start
        in the middle of the main trie. I.e. we ara half way through the main
        trie with another acceptor; that acceptor reaches an end state and then
        we transition to the string acceptor; thus we start string matching in
        the middle of the main trie instead of the root. This can happen e.g.
        if there's tokens in the vocabulary that contain a quote and then
        additional characters afterwards.
        """
        trie_id = id(trie)
        if trie_id in cls._cached_tries:
            return cls._cached_tries[trie_id]
        collapsed_trie = StringCharTokenTrie().from_trie(trie)
        cls._cached_tries[trie_id] = collapsed_trie
        return collapsed_trie

    class Cursor(TokenAcceptor.Cursor):
        """
        Cursor for StringCharAcceptor
        """

        def __init__(self, acceptor, value=None):
            self.acceptor = acceptor
            self.value = value

        def select(self, candidate_chars):
            if self.in_accepted_state():
                return []
            return candidate_chars - StringCharAcceptor.INVALID_CHARS

        def prune(self, trie):
            """
            Use a custom matching trie to avoid an explosion of valid options that
            are equivalent from the point of view of token matching.
            """
            return super().prune(StringCharAcceptor.prepare_trie(trie))

        def advance(self, char):
            if self.in_accepted_state():
                return []
            return [StringCharAcceptor.Cursor(self.acceptor, char)]

        def in_accepted_state(self):
            return self.value is not None

        def get_value(self):
            return self.value


class StringAcceptor(StateMachineAcceptor):
    """
    Accepts a well-formed JSON string
    """

    STATES = [
        [(CharAcceptor('"'), 1)],
        [(CharAcceptor('"'), "$"), (CharAcceptor("\\"), 2), (StringCharAcceptor(), 1)],
        [
            (CharAcceptor('"\\/bfnrt'), 1),
            (CharAcceptor("u"), 3),
        ],
        [(HexDigitAcceptor, 4)],
        [(HexDigitAcceptor, 5)],
        [(HexDigitAcceptor, 6)],
        [(HexDigitAcceptor, 1)],
    ]

    def __init__(self):
        super().__init__(StringAcceptor.STATES)

    class Cursor(StateMachineAcceptor.Cursor):
        """
        Cursor for StringAcceptor
        """

        def __init__(self, acceptor):
            super().__init__(acceptor)
            self.text = ""
            self.length = 0
            self.value = None

        def can_complete_transition(self, transition_value, target_state, is_end_state):
            self.text += transition_value
            if target_state == 1 and self.current_state != 0:
                self.length += 1
            if is_end_state:
                self.value = json.loads(self.text)
            return True

        def get_value(self):
            if self.in_accepted_state():
                return self.value
            else:
                return f"{self.text}👉"


class NumberAcceptor(StateMachineAcceptor):
    """
    Accepts a well-formed JSON number
    """

    STATES = {
        0: [(CharAcceptor("-"), 1), (StateMachineAcceptor.EmptyTransition, 1)],  # Sign
        1: [(CharAcceptor("123456789"), 2), (CharAcceptor("0"), 3)],  # First digit
        2: [
            (DigitAcceptor, 2),
            (StateMachineAcceptor.EmptyTransition, 3),
        ],  # More digits
        3: [(CharAcceptor("."), 4), (StateMachineAcceptor.EmptyTransition, 6)],
        4: [(DigitAcceptor, 5)],  # First decimal
        5: [
            (DigitAcceptor, 5),
            (StateMachineAcceptor.EmptyTransition, 6),
        ],  # More decimals
        6: [(CharAcceptor("eE"), 7)],
        7: [(CharAcceptor("+-"), 8), (StateMachineAcceptor.EmptyTransition, 8)],
        8: [(DigitAcceptor, 9)],  # Exponential, first digit
        9: [(DigitAcceptor, 9)],  # Exponential, more digits
        "$": [2, 3, 5, 9],
    }

    def __init__(self):
        super().__init__(self.STATES, 0, self.STATES["$"])

    class Cursor(StateMachineAcceptor.Cursor):
        """
        Cursor for NumberAcceptor
        """

        def __init__(self, acceptor):
            super().__init__(acceptor)
            self.text = ""
            self.value = None

        def can_complete_transition(self, transition_value, target_state, is_end_state):
            self.text += transition_value
            if is_end_state:
                self.value = json.loads(self.text)
            return True

        def get_value(self):
            if self.value is None:
                return f"{self.text}👉"
            return self.value


class ArrayAcceptor(StateMachineAcceptor):
    """
    Accepts a well-formed JSON array
    """

    def __init__(self):
        super().__init__()

    def get_edges(self, state):
        return {
            0: [(TextAcceptor("["), 1)],
            1: [(WhitespaceAcceptor(), 2), (TextAcceptor("]"), "$")],
            2: [(JsonAcceptor(), 3)],
            3: [(WhitespaceAcceptor(), 4)],
            4: [
                (SequenceAcceptor([TextAcceptor(","), WhitespaceAcceptor()]), 2),
                (TextAcceptor("]"), "$"),
            ],
        }[state]

    class Cursor(StateMachineAcceptor.Cursor):
        """
        Cursor for ArrayAcceptor
        """

        def __init__(self, acceptor):
            super().__init__(acceptor)
            self.value = []

        def can_complete_transition(
            self, transition_value, target_state, is_end_state
        ) -> bool:
            if self.current_state == 3:
                self.value.append(transition_value)
            return True


class ObjectAcceptor(StateMachineAcceptor):
    """
    Accepts a well-formed JSON object
    """

    def __init__(self):
        super().__init__()

    def get_edges(self, state):
        return {
            0: [(TextAcceptor("{"), 1)],
            1: [(WhitespaceAcceptor(), 2), (TextAcceptor("}"), "$")],
            2: [(ObjectAcceptor.PropertyAcceptor(), 3)],
            3: [(WhitespaceAcceptor(), 4)],
            4: [
                (SequenceAcceptor([TextAcceptor(","), WhitespaceAcceptor()]), 2),
                (TextAcceptor("}"), "$"),
            ],
        }[state]

    class Cursor(StateMachineAcceptor.Cursor):
        """
        Cursor for ObjectAcceptor
        """

        def __init__(self, acceptor):
            super().__init__(acceptor)
            self.value = {}

        def can_complete_transition(
            self, transition_value, target_state, is_end_state
        ) -> bool:
            if self.current_state == 2:
                prop_name, prop_value = transition_value
                self.value[prop_name] = prop_value
            return True

        def get_value(self):
            return self.value

    class PropertyAcceptor(SequenceAcceptor):
        """
        JSON object property acceptor
        """

        def __init__(self, graph=None):
            if graph is None:
                graph = [
                    StringAcceptor(),
                    WhitespaceAcceptor(),
                    TextAcceptor(":"),
                    WhitespaceAcceptor(),
                    JsonAcceptor(),
                ]
            super().__init__(graph)

        class Cursor(SequenceAcceptor.Cursor):
            """
            Cursor for ObjectAcceptor.PropertyAcceptor
            """

            def __init__(self, acceptor):
                super().__init__(acceptor)
                self.prop_name = None
                self.prop_value = None

            def can_complete_transition(
                self, transition_value, target_state, is_end_state
            ) -> bool:
                if target_state == 1:
                    self.prop_name = transition_value
                elif is_end_state:
                    self.prop_value = transition_value
                return True

            def get_value(self):
                return (self.prop_name, self.prop_value)


class JsonAcceptor(StateMachineAcceptor):
    """
    Acceptor for JSON input
    """

    def get_edges(self, state):
        if state == 0:
            return [
                (BooleanAcceptor(), "$"),
                (NumberAcceptor(), "$"),
                (StringAcceptor(), "$"),
                (NullAcceptor(), "$"),
                (ObjectAcceptor(), "$"),
                (ArrayAcceptor(), "$"),
            ]
        return []

    class Cursor(StateMachineAcceptor.Cursor):
        """
        Cursor for JsonAcceptor
        """


def prepare_json_acceptor_tries(trie: TokenTrie):
    """
    Pre-cache custom acceptor tries.
    """
    WhitespaceAcceptor.prepare_trie(trie)
    StringCharAcceptor.prepare_trie(trie)
    if '"' in trie.children:
        StringCharAcceptor.prepare_trie(trie.children['"'])
