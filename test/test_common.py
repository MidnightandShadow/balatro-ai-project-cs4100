import pytest
from _pytest.outcomes import fail

from src.common import *


class TestCommon:
    ace_of_spades = Card(Rank.ACE, Suit.SPADES)
    ace_of_clubs = Card(Rank.ACE, Suit.CLUBS)
    king_of_clubs = Card(Rank.KING, Suit.CLUBS)
    king_of_spades = Card(Rank.KING, Suit.SPADES)
    queen_of_diamonds = Card(Rank.QUEEN, Suit.DIAMONDS)
    queen_of_spades = Card(Rank.QUEEN, Suit.SPADES)
    jack_of_hearts = Card(Rank.JACK, Suit.HEARTS)
    jack_of_spades = Card(Rank.JACK, Suit.SPADES)
    ten_of_spades = Card(Rank.TEN, Suit.SPADES)
    ten_of_diamonds = Card(Rank.TEN, Suit.DIAMONDS)
    ten_of_hearts = Card(Rank.TEN, Suit.HEARTS)
    ten_of_clubs = Card(Rank.TEN, Suit.CLUBS)
    nine_of_hearts = Card(Rank.NINE, Suit.HEARTS)
    nine_of_spades = Card(Rank.NINE, Suit.SPADES)
    six_of_spades = Card(Rank.SIX, Suit.SPADES)
    six_of_clubs = Card(Rank.SIX, Suit.CLUBS)
    three_of_spades = Card(Rank.THREE, Suit.SPADES)
    three_of_clubs = Card(Rank.THREE, Suit.CLUBS)
    three_of_diamonds = Card(Rank.THREE, Suit.DIAMONDS)
    three_of_hearts = Card(Rank.THREE, Suit.HEARTS)
    two_of_spades = Card(Rank.TWO, Suit.SPADES)
    flush_hand = [ace_of_spades, three_of_spades, ten_of_spades, queen_of_spades, two_of_spades]
    straight_hand = [king_of_clubs, ace_of_spades, queen_of_diamonds, jack_of_hearts, ten_of_spades]
    scored_straight_hand = ScoredHand(PokerHand.STRAIGHT, straight_hand, [])
    full_house_hand = [three_of_spades, three_of_clubs, three_of_diamonds, ace_of_spades, ace_of_clubs]

    def test_score_card(self):
        assert self.ace_of_spades.score() == 11
        assert (self.king_of_clubs.score() == self.queen_of_diamonds.score() == self.jack_of_hearts.score()
                == self.ten_of_spades.score() == 10)
        assert self.three_of_spades.score() == self.three_of_clubs.score() == 3

    def test_score_hand(self):
        straight_base_chips = ScoringTable.get_chips(PokerHand.STRAIGHT)
        straight_base_mult = ScoringTable.get_mult(PokerHand.STRAIGHT)
        pair_base_chips = ScoringTable.get_chips(PokerHand.PAIR)
        pair_base_mult = ScoringTable.get_mult(PokerHand.PAIR)
        assert self.scored_straight_hand.score() == \
               (straight_base_chips + sum([11, 10, 10, 10, 10])) * straight_base_mult

        assert ScoredHand(PokerHand.PAIR, [self.three_of_spades, self.three_of_clubs],
                          [self.ace_of_spades]).score() == \
               (pair_base_chips + sum([3, 3])) * pair_base_mult

    def test_hand_to_scored_hand_straight(self):
        assert hand_to_scored_hand(self.straight_hand) == self.scored_straight_hand

        not_straight_hand_but_high_card = list(self.straight_hand[0:4])
        not_straight_hand_but_high_card.append(self.nine_of_hearts)
        assert hand_to_scored_hand(not_straight_hand_but_high_card) == \
               (ScoredHand(PokerHand.HIGH_CARD, [self.ace_of_spades],
                           cards_by_rank_descending(not_straight_hand_but_high_card)[1:]))

    def test_hand_to_scored_hand_flush(self):
        assert hand_to_scored_hand(self.flush_hand) == \
               ScoredHand(PokerHand.FLUSH, self.flush_hand, [])

    def test_hand_to_scored_hand_straight_flush(self):
        straight_flush_hand = [self.king_of_spades, self.queen_of_spades, self.jack_of_spades, self.ten_of_spades,
                               self.nine_of_spades]

        assert hand_to_scored_hand(straight_flush_hand) == \
               ScoredHand(PokerHand.STRAIGHT_FLUSH, straight_flush_hand, [])

    def test_hand_to_scored_hand_we_are_counting_royal_flush_as_straight_flush(self):
        royal_flush_hand = [self.ace_of_spades, self.king_of_spades, self.queen_of_spades, self.jack_of_spades,
                            self.ten_of_spades]

        assert hand_to_scored_hand(royal_flush_hand) == \
               ScoredHand(PokerHand.STRAIGHT_FLUSH, royal_flush_hand, [])

    def test_hand_to_scored_hand_full_house(self):
        scored_full_house_hand = ScoredHand(PokerHand.FULL_HOUSE, self.full_house_hand, [])
        assert hand_to_scored_hand(self.full_house_hand) == scored_full_house_hand

        not_full_house_hand = [self.ace_of_spades, self.king_of_clubs,
                               self.three_of_spades, self.three_of_clubs, self.three_of_diamonds]
        assert hand_to_scored_hand(not_full_house_hand) == \
               ScoredHand(PokerHand.THREE_OF_A_KIND,
                          [self.three_of_spades, self.three_of_clubs, self.three_of_diamonds],
                          [self.king_of_clubs, self.ace_of_spades])

    def test_hand_to_scored_hand_four_of_a_kind(self):
        four_of_a_kind_hand = [self.three_of_spades, self.three_of_clubs, self.three_of_diamonds, self.three_of_hearts]
        four_of_a_kind_hand_with_extra_card = four_of_a_kind_hand + [self.ace_of_spades]

        assert hand_to_scored_hand(four_of_a_kind_hand) == \
               ScoredHand(PokerHand.FOUR_OF_A_KIND, four_of_a_kind_hand, [])

        assert hand_to_scored_hand(four_of_a_kind_hand_with_extra_card) == \
               ScoredHand(PokerHand.FOUR_OF_A_KIND, four_of_a_kind_hand, [self.ace_of_spades])

    def test_hand_to_scored_hand_two_pair(self):
        two_pair_hand = [self.three_of_spades, self.three_of_clubs, self.six_of_spades, self.six_of_clubs,
                         self.ten_of_clubs]
        assert hand_to_scored_hand(two_pair_hand) == \
               ScoredHand(PokerHand.TWO_PAIR, two_pair_hand[0:4], [two_pair_hand[4]])

    def test_hand_to_scored_hand_three_of_a_kind(self):
        three_of_a_kind_hand = [self.three_of_spades, self.three_of_clubs, self.three_of_hearts, self.six_of_clubs]
        assert hand_to_scored_hand(three_of_a_kind_hand) == \
               ScoredHand(PokerHand.THREE_OF_A_KIND, three_of_a_kind_hand[0:3], [three_of_a_kind_hand[3]])

    def test_hand_to_scored_hand_pair(self):
        pair_hand = [self.three_of_spades, self.three_of_clubs, self.six_of_spades, self.ten_of_clubs]
        assert hand_to_scored_hand(pair_hand) == \
               ScoredHand(PokerHand.PAIR, pair_hand[0:2], pair_hand[2:4])

    def test_hand_to_scored_hand_high_card(self):
        high_card_hand = [self.three_of_spades, self.jack_of_hearts, self.ace_of_spades, self.king_of_clubs,
                          self.two_of_spades]
        assert hand_to_scored_hand(high_card_hand) == \
               ScoredHand(PokerHand.HIGH_CARD, [self.ace_of_spades],
                          [self.king_of_clubs, self.jack_of_hearts, self.three_of_spades,
                          self.two_of_spades])

    def test_group_cards_by_attribute(self):
        assert group_cards_by_attribute(self.full_house_hand, CardAttribute.RANK) == \
               [[self.three_of_spades, self.three_of_clubs, self.three_of_diamonds],
                [self.ace_of_spades, self.ace_of_clubs]]

        assert group_cards_by_attribute(self.full_house_hand, CardAttribute.SUIT) == \
               [[self.three_of_spades, self.ace_of_spades],
                [self.three_of_clubs, self.ace_of_clubs],
                [self.three_of_diamonds]]

        six_cards = [self.ten_of_spades, self.ten_of_diamonds, self.ten_of_hearts, self.ten_of_clubs,
                     self.two_of_spades, self.six_of_clubs]

        assert group_cards_by_attribute(six_cards, CardAttribute.SUIT) == \
               [[self.ten_of_spades, self.two_of_spades], [self.ten_of_clubs, self.six_of_clubs],
                [self.ten_of_hearts], [self.ten_of_diamonds]]

        all_one_suit = [self.ten_of_spades, self.two_of_spades]
        assert group_cards_by_attribute(all_one_suit, CardAttribute.SUIT) == \
               [[self.ten_of_spades, self.two_of_spades]]

        six_cards_all_one_suit = [self.ace_of_spades, self.queen_of_spades, self.ten_of_spades, self.two_of_spades,
                                  self.three_of_spades, self.six_of_spades]

        assert group_cards_by_attribute(six_cards_all_one_suit, CardAttribute.SUIT) == \
               [six_cards_all_one_suit]
