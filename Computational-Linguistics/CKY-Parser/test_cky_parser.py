import unittest
import nltk
from cky_parser import *

class TestCKYParser(unittest.TestCase):

    def test_num_parses(self):
        toy_gram = nltk.CFG.fromstring("""
            S -> A B
            S -> B C
            A -> B A
            B -> C C
            C -> A B
            A -> 'a'
            B -> 'b'
            C -> 'a'
            """)
        self.assertEqual(cky_parser(['b', 'a', 'a', 'b', 'a'], toy_gram, parser=True), 2)

    def test_ungrammaticalSent_missingGrammar(self):
        grammar = nltk.CFG.fromstring("""
            S -> NP VP
            PP -> P NP | P N
            NP -> Det N | Det NPP | 'I' | 'John'
            NPP -> N PP | N P 
            VP -> V NP | VP PP
            Det -> 'an' | 'my' | 'a'
            N -> 'elephant' | 'pajamas' | 'shot' | 'cake'
            V -> 'shot' | 'ate'
            P -> 'in'
            """)
        self.assertEqual(cky_parser('I shot an elephant in my pajamas'.split(), grammar, True), 2)
        self.assertEqual(cky_parser('Mary shot an elephant in my pajamas'.split(), grammar, True), 0)
        self.assertEqual(cky_parser('I shot an my pajamas'.split(), grammar, True), 0)

if __name__ == '__main__':
    unittest.main()