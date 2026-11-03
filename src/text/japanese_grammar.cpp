// Adapted from Open JTalk: Copyright (c) 2008-2016 Nagoya Institute of Technology.
// BSD-3-Clause; see licenses/linguistics.txt for the full notice and terms.
#include "text/japanese_grammar.h"
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
namespace tts::text::japanese {
namespace {
int strtopcmp(const char *text, const char *pattern) {
    auto size = std::strlen(pattern);
    return std::strncmp(text, pattern, size) == 0 ? int(size) : -1;
}
Node *insert(Node *previous, Node *next, Node *node) {
    if (!previous || !next || !node) throw std::logic_error("invalid Japanese node insertion");
    auto *tail = node;
    while (tail->next) tail = tail->next;
    previous->next = node; node->prev = previous;
    next->prev = tail; tail->next = next;
    return tail;
}
std::string accent_rule(std::string_view text, std::string_view previous, int &addition) {
    size_t cursor = 0;
    auto token = [&] {
        auto end = text.find_first_of("%@/", cursor);
        auto value = text.substr(cursor, end == text.npos ? text.size() - cursor : end - cursor);
        char separator = end == text.npos ? '\0' : text[end];
        cursor = end == text.npos ? text.size() : end + 1;
        return std::pair{std::string(value), separator};
    };
    char separator = ' ';
    while (separator) {
        auto item = token(); separator = item.second;
        if ((separator == '%' && previous.find(item.first) != previous.npos) || separator == '@' || separator == '/' || !separator) {
            if (separator == '%') { item = token(); separator = item.second; }
            addition = separator == '@' ? std::atoi(token().first.c_str()) : 0;
            return item.first;
        }
        while (separator == '%' || separator == '@') separator = token().second;
    }
    addition = 0;
    return "*";
}
struct Process {
    Document &document;
    const Rules &rules;

void set_pronunciation(Document * njd)
{
   const auto *NJD_SET_PRONUNCIATION_CHOUON = rules.constant("NJD_SET_PRONUNCIATION_CHOUON");
   const auto *NJD_SET_PRONUNCIATION_DESU_PRON = rules.constant("NJD_SET_PRONUNCIATION_DESU_PRON");
   const auto *NJD_SET_PRONUNCIATION_DESU_STR = rules.constant("NJD_SET_PRONUNCIATION_DESU_STR");
   const auto *NJD_SET_PRONUNCIATION_DOUSHI = rules.constant("NJD_SET_PRONUNCIATION_DOUSHI");
   const auto *NJD_SET_PRONUNCIATION_FILLER = rules.constant("NJD_SET_PRONUNCIATION_FILLER");
   const auto *NJD_SET_PRONUNCIATION_JODOUSHI = rules.constant("NJD_SET_PRONUNCIATION_JODOUSHI");
   const auto *NJD_SET_PRONUNCIATION_KIGOU = rules.constant("NJD_SET_PRONUNCIATION_KIGOU");
   const auto *NJD_SET_PRONUNCIATION_MASU_PRON = rules.constant("NJD_SET_PRONUNCIATION_MASU_PRON");
   const auto *NJD_SET_PRONUNCIATION_MASU_STR = rules.constant("NJD_SET_PRONUNCIATION_MASU_STR");
   const auto *NJD_SET_PRONUNCIATION_QUESTION = rules.constant("NJD_SET_PRONUNCIATION_QUESTION");
   const auto *NJD_SET_PRONUNCIATION_TOUTEN = rules.constant("NJD_SET_PRONUNCIATION_TOUTEN");
   const auto *NJD_SET_PRONUNCIATION_U = rules.constant("NJD_SET_PRONUNCIATION_U");
   const auto *njd_set_pronunciation_list = rules.table("njd_set_pronunciation_list").data();
   const auto *njd_set_pronunciation_symbol_list = rules.table("njd_set_pronunciation_symbol_list").data();
   Node *node;
   const char *str;
   int i, j = 0;
   int pos;
   int len;
   for (node = njd->head; node != nullptr; node = node->next) {
      if (node->mora_size == 0) {
         assign(node->read, nullptr);
         assign(node->pron, nullptr);
         {
            str = get(node->string);
            len = strlen(str);
            for (pos = 0; pos < len;) {
               for (i = 0, j = 0; njd_set_pronunciation_list[i] != nullptr; i += 3) {
                  j = strtopcmp(&str[pos], njd_set_pronunciation_list[i]);
                  if (j > 0)
                     break;
               }
               if (j > 0) {
                  pos += j;
                  node->read += njd_set_pronunciation_list[i + 1];
                  node->pron += njd_set_pronunciation_list[i + 1];
                  node->mora_size += atoi(njd_set_pronunciation_list[i + 2]);
               } else {
                  pos++;
               }
            }
            if (node->mora_size != 0) {
               assign(node->pos, NJD_SET_PRONUNCIATION_FILLER);
               assign(node->pos_group1, nullptr);
               assign(node->pos_group2, nullptr);
               assign(node->pos_group3, nullptr);
            }
            if (strcmp(get(node->orig), "*") == 0) {
               assign(node->orig, str);
            }
         }
         if (strcmp(get(node->pron), "*") == 0) {
            for (i = 0; njd_set_pronunciation_symbol_list[i] != nullptr; i += 2) {
               if (strcmp(get(node->string), njd_set_pronunciation_symbol_list[i]) == 0) {
                  assign(node->read, njd_set_pronunciation_symbol_list[i + 1]);
                  assign(node->pron, njd_set_pronunciation_symbol_list[i + 1]);
                  break;
               }
            }
         }
         if (strcmp(get(node->pron), "*") == 0) {
            assign(node->read, NJD_SET_PRONUNCIATION_TOUTEN);
            assign(node->pron, NJD_SET_PRONUNCIATION_TOUTEN);
            assign(node->pos, NJD_SET_PRONUNCIATION_KIGOU);
         }
      }
   }
   njd->remove_silent();
   {
      Node *head_of_kana_filler_sequence = nullptr;
      int find;
      for (node = njd->head; node != nullptr; node = node->next) {
         if (strcmp(get(node->pos), NJD_SET_PRONUNCIATION_FILLER) == 0) {
            find = 0;
            for (i = 0; njd_set_pronunciation_list[i] != nullptr; i += 3) {
               if (strcmp(get(node->string), njd_set_pronunciation_list[i]) == 0) {
                  find = 1;
                  if (head_of_kana_filler_sequence == nullptr) {
                     head_of_kana_filler_sequence = node;
                  } else {
                     head_of_kana_filler_sequence->string += get(node->string);
                     head_of_kana_filler_sequence->orig += get(node->orig);
                     head_of_kana_filler_sequence->read += get(node->read);
                     head_of_kana_filler_sequence->pron += get(node->pron);
                     head_of_kana_filler_sequence->mora_size += node->mora_size;
                     assign(node->pron, nullptr);
                  }
                  break;
               }
            }
            if (find == 0) {
               head_of_kana_filler_sequence = nullptr;
            }
         } else {
            head_of_kana_filler_sequence = nullptr;
         }
      }
   }
   njd->remove_silent();
   for (node = njd->head; node != nullptr; node = node->next) {
      if (node->next != nullptr
          && strcmp(get(node->next->pron), NJD_SET_PRONUNCIATION_U) == 0
          && strcmp(get(node->next->pos), NJD_SET_PRONUNCIATION_JODOUSHI) == 0
          && (strcmp(get(node->pos), NJD_SET_PRONUNCIATION_DOUSHI) == 0
              || strcmp(get(node->pos), NJD_SET_PRONUNCIATION_JODOUSHI) == 0)
          && node->mora_size > 0) {
         assign(node->next->pron, NJD_SET_PRONUNCIATION_CHOUON);
      }
      if (node->next != nullptr
          && strcmp(get(node->pos), NJD_SET_PRONUNCIATION_JODOUSHI) == 0
          && strcmp(get(node->next->string), NJD_SET_PRONUNCIATION_QUESTION) == 0) {
         if (strcmp(get(node->string), NJD_SET_PRONUNCIATION_DESU_STR) == 0)
            assign(node->pron, NJD_SET_PRONUNCIATION_DESU_PRON);
         else if (strcmp(get(node->string), NJD_SET_PRONUNCIATION_MASU_STR) == 0)
            assign(node->pron, NJD_SET_PRONUNCIATION_MASU_PRON);
      }
   }
}


int get_digit(Node * node, int convert_flag)
{
   const auto *NJD_SET_DIGIT_KAZU = rules.constant("NJD_SET_DIGIT_KAZU");
   const auto *njd_set_digit_rule_numeral_list1 = rules.table("njd_set_digit_rule_numeral_list1").data();
   int i;
   if (strcmp(get(node->string), "*") == 0)
      return -1;
   if (strcmp(get(node->pos_group1), NJD_SET_DIGIT_KAZU) == 0)
      for (i = 0; njd_set_digit_rule_numeral_list1[i] != nullptr; i += 3)
         if (strcmp(njd_set_digit_rule_numeral_list1[i], get(node->string)) == 0) {
            if (convert_flag == 1) {
               assign(node->string, njd_set_digit_rule_numeral_list1[i + 2]);
               assign(node->orig, njd_set_digit_rule_numeral_list1[i + 2]);
            }
            return atoi(njd_set_digit_rule_numeral_list1[i + 1]);
         }
   return -1;
}
int is_period(const char *str)
{
   const auto *NJD_SET_DIGIT_TEN1 = rules.constant("NJD_SET_DIGIT_TEN1");
   const auto *NJD_SET_DIGIT_TEN2 = rules.constant("NJD_SET_DIGIT_TEN2");
   if (str != nullptr &&
       (strcmp(str, NJD_SET_DIGIT_TEN1) == 0 || strcmp(str, NJD_SET_DIGIT_TEN2) == 0)) {
      return 1;
   } else {
      return 0;
   }
}
int is_comma(const char *str)
{
   const auto *NJD_SET_DIGIT_COMMA = rules.constant("NJD_SET_DIGIT_COMMA");
   if (str != nullptr && strcmp(str, NJD_SET_DIGIT_COMMA) == 0) {
      return 1;
   } else {
      return 0;
   }
}
int get_digit_sequence_score(Node * start, Node * end)
{
   const auto *NJD_SET_DIGIT_BANGOU = rules.constant("NJD_SET_DIGIT_BANGOU");
   const auto *NJD_SET_DIGIT_FUKUSHIKANOU = rules.constant("NJD_SET_DIGIT_FUKUSHIKANOU");
   const auto *NJD_SET_DIGIT_HAIHUN1 = rules.constant("NJD_SET_DIGIT_HAIHUN1");
   const auto *NJD_SET_DIGIT_HAIHUN2 = rules.constant("NJD_SET_DIGIT_HAIHUN2");
   const auto *NJD_SET_DIGIT_HAIHUN3 = rules.constant("NJD_SET_DIGIT_HAIHUN3");
   const auto *NJD_SET_DIGIT_HAIHUN4 = rules.constant("NJD_SET_DIGIT_HAIHUN4");
   const auto *NJD_SET_DIGIT_HAIHUN5 = rules.constant("NJD_SET_DIGIT_HAIHUN5");
   const auto *NJD_SET_DIGIT_JOSUUSHI = rules.constant("NJD_SET_DIGIT_JOSUUSHI");
   const auto *NJD_SET_DIGIT_KAKKO1 = rules.constant("NJD_SET_DIGIT_KAKKO1");
   const auto *NJD_SET_DIGIT_KAKKO2 = rules.constant("NJD_SET_DIGIT_KAKKO2");
   const auto *NJD_SET_DIGIT_KAZU = rules.constant("NJD_SET_DIGIT_KAZU");
   const auto *NJD_SET_DIGIT_SUUSETSUZOKU = rules.constant("NJD_SET_DIGIT_SUUSETSUZOKU");
   const char *buff_pos_group1 = nullptr;
   const char *buff_pos_group2 = nullptr;
   const char *buff_string = nullptr;
   int score = 0;
   if (start->prev) {
      buff_pos_group1 = get(start->prev->pos_group1);
      buff_pos_group2 = get(start->prev->pos_group2);
      buff_string = get(start->prev->string);
      if (strcmp(buff_pos_group1, NJD_SET_DIGIT_SUUSETSUZOKU) == 0)
         score += 2;
      if (strcmp(buff_pos_group2, NJD_SET_DIGIT_JOSUUSHI) == 0 || strcmp(buff_pos_group1, NJD_SET_DIGIT_FUKUSHIKANOU) == 0)
         score += 1;
      if (buff_string != nullptr) {
         if (is_period(buff_string) == 1) {
            if (!start->prev->prev
                || strcmp(get(start->prev->prev->pos_group1), NJD_SET_DIGIT_KAZU) != 0)
               score += 0;
            else
               score -= 5;
         } else if (strcmp(buff_string, NJD_SET_DIGIT_HAIHUN1) == 0)
            score -= 2;
         else if (strcmp(buff_string, NJD_SET_DIGIT_HAIHUN2) == 0)
            score -= 2;
         else if (strcmp(buff_string, NJD_SET_DIGIT_HAIHUN3) == 0)
            score -= 2;
         else if (strcmp(buff_string, NJD_SET_DIGIT_HAIHUN4) == 0)
            score -= 2;
         else if (strcmp(buff_string, NJD_SET_DIGIT_HAIHUN5) == 0)
            score -= 2;
         else if (strcmp(buff_string, NJD_SET_DIGIT_KAKKO1) == 0) {
            if (!start->prev->prev
                || strcmp(get(start->prev->prev->pos_group1), NJD_SET_DIGIT_KAZU) != 0)
               score += 0;
            else
               score -= 2;
         } else if (strcmp(buff_string, NJD_SET_DIGIT_KAKKO2) == 0)
            score -= 2;
         else if (strcmp(buff_string, NJD_SET_DIGIT_BANGOU) == 0)
            score -= 2;
      }
      if (start->prev->prev) {
         buff_string = get(start->prev->prev->string);
         if (strcmp(buff_string, NJD_SET_DIGIT_BANGOU) == 0)
            score -= 2;
      }
   }
   if (end->next) {
      buff_pos_group1 = get(end->next->pos_group1);
      buff_pos_group2 = get(end->next->pos_group2);
      buff_string = get(end->next->string);
      if (strcmp(buff_pos_group2, NJD_SET_DIGIT_JOSUUSHI) == 0
          || strcmp(buff_pos_group1, NJD_SET_DIGIT_FUKUSHIKANOU) == 0)
         score += 2;
      if (buff_string != nullptr) {
         if (strcmp(buff_string, NJD_SET_DIGIT_HAIHUN1) == 0)
            score -= 2;
         else if (strcmp(buff_string, NJD_SET_DIGIT_HAIHUN2) == 0)
            score -= 2;
         else if (strcmp(buff_string, NJD_SET_DIGIT_HAIHUN3) == 0)
            score -= 2;
         else if (strcmp(buff_string, NJD_SET_DIGIT_HAIHUN4) == 0)
            score -= 2;
         else if (strcmp(buff_string, NJD_SET_DIGIT_HAIHUN5) == 0)
            score -= 2;
         else if (strcmp(buff_string, NJD_SET_DIGIT_KAKKO1) == 0)
            score -= 2;
         else if (strcmp(buff_string, NJD_SET_DIGIT_KAKKO2) == 0) {
            if (!end->next->next
                || strcmp(get(end->next->next->pos_group1), NJD_SET_DIGIT_KAZU) != 0)
               score += 0;
            else
               score -= 2;
         } else if (strcmp(buff_string, NJD_SET_DIGIT_BANGOU) == 0)
            score -= 2;
         else if (is_period(buff_string) == 1)
            score += 4;
      }
   }
   return score;
}
void convert_digit_sequence_for_non_numerical_reading(Node * start, Node * end)
{
   const auto *NJD_SET_DIGIT_FIVE = rules.constant("NJD_SET_DIGIT_FIVE");
   const auto *NJD_SET_DIGIT_FIVE_AFTER_DP = rules.constant("NJD_SET_DIGIT_FIVE_AFTER_DP");
   const auto *NJD_SET_DIGIT_TWO = rules.constant("NJD_SET_DIGIT_TWO");
   const auto *NJD_SET_DIGIT_TWO_AFTER_DP = rules.constant("NJD_SET_DIGIT_TWO_AFTER_DP");
   const auto *NJD_SET_DIGIT_ZERO1 = rules.constant("NJD_SET_DIGIT_ZERO1");
   const auto *NJD_SET_DIGIT_ZERO2 = rules.constant("NJD_SET_DIGIT_ZERO2");
   const auto *NJD_SET_DIGIT_ZERO_AFTER_DP = rules.constant("NJD_SET_DIGIT_ZERO_AFTER_DP");
   Node *node;
   int size = 0;
   for (node = start; node != end->next; node = node->next)
      size++;
   if (size <= 1)
      return;
   for (node = start, size = 0; node != end->next; node = node->next) {
      if (strcmp(get(node->string), NJD_SET_DIGIT_ZERO1) == 0
          || strcmp(get(node->string), NJD_SET_DIGIT_ZERO2) == 0) {
         assign(node->pron, NJD_SET_DIGIT_ZERO_AFTER_DP);
         node->mora_size = 2;
      } else if (strcmp(get(node->string), NJD_SET_DIGIT_TWO) == 0) {
         assign(node->pron, NJD_SET_DIGIT_TWO_AFTER_DP);
         node->mora_size = 2;
      } else if (strcmp(get(node->string), NJD_SET_DIGIT_FIVE) == 0) {
         assign(node->pron, NJD_SET_DIGIT_FIVE_AFTER_DP);
         node->mora_size = 2;
      }
      assign(node->chain_rule, nullptr);
      if (size % 2 == 0) {
         node->chain_flag = 0;
      } else {
         node->chain_flag = 1;
         node->prev->acc = 3;
      }
      size++;
   }
}
void convert_digit_sequence_for_numerical_reading(Node * start, Node * end)
{
   const auto *njd_set_digit_rule_numeral_list2 = rules.table("njd_set_digit_rule_numeral_list2").data();
   const auto *njd_set_digit_rule_numeral_list3 = rules.table("njd_set_digit_rule_numeral_list3").data();
   Node *node;
   Node *newnode;
   int digit;
   int place = 0;
   int index;
   int size = 0;
   int have = 0;
   for (node = start; node != end->next; node = node->next)
      size++;
   if (size <= 1)
      return;
   index = size % 4;
   if (index == 0)
      index = 4;
   if (size > index)
      place = (size - index) / 4;
   index--;
   if (place > 17)
      return;
   for (node = start; node != end->next; node = node->next) {
      digit = get_digit(node, 0);
      if (index == 0) {
         if (digit == 0) {
            assign(node->pron, nullptr);
            node->acc = 0;
            node->mora_size = 0;
         } else {
            have = 1;
         }
         if (have == 1) {
            if (place > 0) {
               newnode = document.create();
               document.load(newnode, njd_set_digit_rule_numeral_list3[place]);
               node = insert(node, node->next, newnode);
            }
            have = 0;
         }
         place--;
      } else {
         if (digit <= 0) {
            assign(node->pron, nullptr);
            node->acc = 0;
            node->mora_size = 0;
         } else if (digit == 1) {
            document.load(node, njd_set_digit_rule_numeral_list2[index]);
            have = 1;
         } else {
            newnode = document.create();
            document.load(newnode, njd_set_digit_rule_numeral_list2[index]);
            node = insert(node, node->next, newnode);
            have = 1;
         }
      }
      index--;
      if (index < 0)
         index = 4 - 1;
   }
}
int search_numerative_class(const char *const list[], Node * node)
{
   int i;
   const char *str = get(node->string);
   if (strcmp(str, "*") == 0)
      return 0;
   for (i = 0; list[i] != nullptr; i++) {
      if (strcmp(list[i], str) == 0)
         return 1;
   }
   return 0;
}
void convert_digit_pron(const char *const list[], Node * node)
{
   int i;
   const char *str = get(node->string);
   if (strcmp(str, "*") == 0)
      return;
   for (i = 0; list[i] != nullptr; i += 4) {
      if (strcmp(list[i], str) == 0) {
         assign(node->pron, list[i + 1]);
         node->acc = std::max(0, atoi(list[i + 2]));
         node->mora_size = std::max(0, atoi(list[i + 3]));
         return;
      }
   }
}
void convert_numerative_pron(const char *const list[], Node * node1, Node * node2)
{
   const auto *njd_set_digit_rule_semivoiced_sound_symbol_list = rules.table("njd_set_digit_rule_semivoiced_sound_symbol_list").data();
   const auto *njd_set_digit_rule_voiced_sound_symbol_list = rules.table("njd_set_digit_rule_voiced_sound_symbol_list").data();
   int i, j;
   int type = 0;
   const char *str = get(node1->string);
   std::string buff;
   if (strcmp(str, "*") == 0)
      return;
   for (i = 0; list[i] != nullptr; i += 2) {
      if (strcmp(list[i], str) == 0) {
         type = atoi(list[i + 1]);
         break;
      }
   }
   if (type == 1) {
      for (i = 0; njd_set_digit_rule_voiced_sound_symbol_list[i] != nullptr; i += 2) {
         str = get(node2->pron);
         j = strtopcmp(str, njd_set_digit_rule_voiced_sound_symbol_list[i]);
         if (j >= 0) {
            buff = njd_set_digit_rule_voiced_sound_symbol_list[i + 1];
            buff += &str[j];
            assign(node2->pron, buff.c_str());
            break;
         }
      }
   } else if (type == 2) {
      for (i = 0; njd_set_digit_rule_semivoiced_sound_symbol_list[i] != nullptr; i += 2) {
         str = get(node2->pron);
         j = strtopcmp(str, njd_set_digit_rule_semivoiced_sound_symbol_list[i]);
         if (j >= 0) {
            buff = njd_set_digit_rule_semivoiced_sound_symbol_list[i + 1];
            buff += &str[j];
            assign(node2->pron, buff.c_str());
            break;
         }
      }
   }
}
void convert_digit_sequence(Document * njd, Node * s, Node * e)
{
   Node *node;
   Node *final_digit_before_period = s;
   int numerical_reading = 1;
   int num_comma = 0;
   Node *first_comma_before_period = nullptr;
   if (s == nullptr || e == nullptr) {
      return;
   }
   if (is_comma(get(s->string)) == 1 || is_period(get(s->string)) == 1) {
      if (s != e)
         convert_digit_sequence(njd, s->next, e);
      return;
   }
   while (final_digit_before_period->next != nullptr && final_digit_before_period != e &&
          is_period(get(final_digit_before_period->next->string)) != 1) {
      final_digit_before_period = final_digit_before_period->next;
   }
   while (final_digit_before_period->prev != nullptr && final_digit_before_period != s &&
          is_comma(get(final_digit_before_period->string)) == 1) {
      final_digit_before_period = final_digit_before_period->prev;
   }
   {
      int rindex = 0;
      for (rindex = 0, node = final_digit_before_period;; node = node->prev, rindex++) {
         if (is_comma(get(node->string)) == 1) {
            first_comma_before_period = node;
            num_comma++;
            if (numerical_reading == 1 && rindex % 4 != 3) {
               numerical_reading = 0;
            }
         } else if (numerical_reading == 1 && rindex % 4 == 3) {
            numerical_reading = 0;
         }
         if (node == s) {
            break;
         }
      }
   }
   if (s != final_digit_before_period && get_digit(s, 0) == 0)
      numerical_reading = -1;
   if (numerical_reading == 1 && num_comma == 0)
      numerical_reading = 0;
   if (numerical_reading == 1) {
      if (num_comma > 0) {
         for (node = s; node != final_digit_before_period;) {
            if (is_comma(get(node->string)) == 1)
               node = njd->remove(node);
            else
               node = node->next;
         }
      }
      convert_digit_sequence_for_numerical_reading(s, final_digit_before_period);
      if (final_digit_before_period != e)
         convert_digit_sequence(njd, final_digit_before_period->next, e);
   } else {
      Node *final_digit;
      if (first_comma_before_period == nullptr)
         final_digit = final_digit_before_period;
      else
         final_digit = first_comma_before_period->prev;
      if (numerical_reading == 0) {
         if (get_digit_sequence_score(s, final_digit) >= 0)
            numerical_reading = 1;
         else
            numerical_reading = -1;
      }
      if (numerical_reading == 1) {
         convert_digit_sequence_for_numerical_reading(s, final_digit);
      } else {
         convert_digit_sequence_for_non_numerical_reading(s, final_digit);
      }
      if (final_digit != e)
         convert_digit_sequence(njd, final_digit->next, e);
   }
}
void set_digit(Document * njd)
{
   const auto *NJD_SET_DIGIT_FIVE = rules.constant("NJD_SET_DIGIT_FIVE");
   const auto *NJD_SET_DIGIT_FIVE_BEFORE_DP = rules.constant("NJD_SET_DIGIT_FIVE_BEFORE_DP");
   const auto *NJD_SET_DIGIT_FOUR = rules.constant("NJD_SET_DIGIT_FOUR");
   const auto *NJD_SET_DIGIT_FUKUSHIKANOU = rules.constant("NJD_SET_DIGIT_FUKUSHIKANOU");
   const auto *NJD_SET_DIGIT_GATSU = rules.constant("NJD_SET_DIGIT_GATSU");
   const auto *NJD_SET_DIGIT_HATSUKAKAN = rules.constant("NJD_SET_DIGIT_HATSUKAKAN");
   const auto *NJD_SET_DIGIT_JOSUUSHI = rules.constant("NJD_SET_DIGIT_JOSUUSHI");
   const auto *NJD_SET_DIGIT_JUYOKKA = rules.constant("NJD_SET_DIGIT_JUYOKKA");
   const auto *NJD_SET_DIGIT_JUYOKKAKAN = rules.constant("NJD_SET_DIGIT_JUYOKKAKAN");
   const auto *NJD_SET_DIGIT_KAZU = rules.constant("NJD_SET_DIGIT_KAZU");
   const auto *NJD_SET_DIGIT_KIGOU = rules.constant("NJD_SET_DIGIT_KIGOU");
   const auto *NJD_SET_DIGIT_MEISHI = rules.constant("NJD_SET_DIGIT_MEISHI");
   const auto *NJD_SET_DIGIT_NICHI = rules.constant("NJD_SET_DIGIT_NICHI");
   const auto *NJD_SET_DIGIT_NICHIKAN = rules.constant("NJD_SET_DIGIT_NICHIKAN");
   const auto *NJD_SET_DIGIT_NIJU = rules.constant("NJD_SET_DIGIT_NIJU");
   const auto *NJD_SET_DIGIT_NIN = rules.constant("NJD_SET_DIGIT_NIN");
   const auto *NJD_SET_DIGIT_ONE = rules.constant("NJD_SET_DIGIT_ONE");
   const auto *NJD_SET_DIGIT_SIX = rules.constant("NJD_SET_DIGIT_SIX");
   const auto *NJD_SET_DIGIT_TEN = rules.constant("NJD_SET_DIGIT_TEN");
   const auto *NJD_SET_DIGIT_TEN_FEATURE = rules.constant("NJD_SET_DIGIT_TEN_FEATURE");
   const auto *NJD_SET_DIGIT_TSUITACHI = rules.constant("NJD_SET_DIGIT_TSUITACHI");
   const auto *NJD_SET_DIGIT_TWO = rules.constant("NJD_SET_DIGIT_TWO");
   const auto *NJD_SET_DIGIT_TWO_BEFORE_DP = rules.constant("NJD_SET_DIGIT_TWO_BEFORE_DP");
   const auto *NJD_SET_DIGIT_YOKKAKAN = rules.constant("NJD_SET_DIGIT_YOKKAKAN");
   const auto *NJD_SET_DIGIT_ZERO1 = rules.constant("NJD_SET_DIGIT_ZERO1");
   const auto *NJD_SET_DIGIT_ZERO2 = rules.constant("NJD_SET_DIGIT_ZERO2");
   const auto *NJD_SET_DIGIT_ZERO_BEFORE_DP = rules.constant("NJD_SET_DIGIT_ZERO_BEFORE_DP");
   const auto *NJD_SET_DITIT_HATSUKA = rules.constant("NJD_SET_DITIT_HATSUKA");
   const auto *NJD_SET_DITIT_YOKKA = rules.constant("NJD_SET_DITIT_YOKKA");
   const auto *njd_set_digit_rule_conv_table1b = rules.table("njd_set_digit_rule_conv_table1b").data();
   const auto *njd_set_digit_rule_conv_table1c1 = rules.table("njd_set_digit_rule_conv_table1c1").data();
   const auto *njd_set_digit_rule_conv_table1c2 = rules.table("njd_set_digit_rule_conv_table1c2").data();
   const auto *njd_set_digit_rule_conv_table1d = rules.table("njd_set_digit_rule_conv_table1d").data();
   const auto *njd_set_digit_rule_conv_table1e = rules.table("njd_set_digit_rule_conv_table1e").data();
   const auto *njd_set_digit_rule_conv_table1f = rules.table("njd_set_digit_rule_conv_table1f").data();
   const auto *njd_set_digit_rule_conv_table1g = rules.table("njd_set_digit_rule_conv_table1g").data();
   const auto *njd_set_digit_rule_conv_table1h = rules.table("njd_set_digit_rule_conv_table1h").data();
   const auto *njd_set_digit_rule_conv_table1i = rules.table("njd_set_digit_rule_conv_table1i").data();
   const auto *njd_set_digit_rule_conv_table1j = rules.table("njd_set_digit_rule_conv_table1j").data();
   const auto *njd_set_digit_rule_conv_table1k = rules.table("njd_set_digit_rule_conv_table1k").data();
   const auto *njd_set_digit_rule_conv_table2b = rules.table("njd_set_digit_rule_conv_table2b").data();
   const auto *njd_set_digit_rule_conv_table2c = rules.table("njd_set_digit_rule_conv_table2c").data();
   const auto *njd_set_digit_rule_conv_table2d = rules.table("njd_set_digit_rule_conv_table2d").data();
   const auto *njd_set_digit_rule_conv_table2e = rules.table("njd_set_digit_rule_conv_table2e").data();
   const auto *njd_set_digit_rule_conv_table2f = rules.table("njd_set_digit_rule_conv_table2f").data();
   const auto *njd_set_digit_rule_conv_table3 = rules.table("njd_set_digit_rule_conv_table3").data();
   const auto *njd_set_digit_rule_conv_table4 = rules.table("njd_set_digit_rule_conv_table4").data();
   const auto *njd_set_digit_rule_conv_table5 = rules.table("njd_set_digit_rule_conv_table5").data();
   const auto *njd_set_digit_rule_conv_table6 = rules.table("njd_set_digit_rule_conv_table6").data();
   const auto *njd_set_digit_rule_numeral_list10 = rules.table("njd_set_digit_rule_numeral_list10").data();
   const auto *njd_set_digit_rule_numeral_list11 = rules.table("njd_set_digit_rule_numeral_list11").data();
   const auto *njd_set_digit_rule_numeral_list4 = rules.table("njd_set_digit_rule_numeral_list4").data();
   const auto *njd_set_digit_rule_numeral_list5 = rules.table("njd_set_digit_rule_numeral_list5").data();
   const auto *njd_set_digit_rule_numeral_list6 = rules.table("njd_set_digit_rule_numeral_list6").data();
   const auto *njd_set_digit_rule_numeral_list7 = rules.table("njd_set_digit_rule_numeral_list7").data();
   const auto *njd_set_digit_rule_numeral_list8 = rules.table("njd_set_digit_rule_numeral_list8").data();
   const auto *njd_set_digit_rule_numeral_list9 = rules.table("njd_set_digit_rule_numeral_list9").data();
   const auto *njd_set_digit_rule_numerative_class1b = rules.table("njd_set_digit_rule_numerative_class1b").data();
   const auto *njd_set_digit_rule_numerative_class1c1 = rules.table("njd_set_digit_rule_numerative_class1c1").data();
   const auto *njd_set_digit_rule_numerative_class1c2 = rules.table("njd_set_digit_rule_numerative_class1c2").data();
   const auto *njd_set_digit_rule_numerative_class1d = rules.table("njd_set_digit_rule_numerative_class1d").data();
   const auto *njd_set_digit_rule_numerative_class1e = rules.table("njd_set_digit_rule_numerative_class1e").data();
   const auto *njd_set_digit_rule_numerative_class1f = rules.table("njd_set_digit_rule_numerative_class1f").data();
   const auto *njd_set_digit_rule_numerative_class1g = rules.table("njd_set_digit_rule_numerative_class1g").data();
   const auto *njd_set_digit_rule_numerative_class1h = rules.table("njd_set_digit_rule_numerative_class1h").data();
   const auto *njd_set_digit_rule_numerative_class1i = rules.table("njd_set_digit_rule_numerative_class1i").data();
   const auto *njd_set_digit_rule_numerative_class1j = rules.table("njd_set_digit_rule_numerative_class1j").data();
   const auto *njd_set_digit_rule_numerative_class1k = rules.table("njd_set_digit_rule_numerative_class1k").data();
   const auto *njd_set_digit_rule_numerative_class2b = rules.table("njd_set_digit_rule_numerative_class2b").data();
   const auto *njd_set_digit_rule_numerative_class2c = rules.table("njd_set_digit_rule_numerative_class2c").data();
   const auto *njd_set_digit_rule_numerative_class2d = rules.table("njd_set_digit_rule_numerative_class2d").data();
   const auto *njd_set_digit_rule_numerative_class2e = rules.table("njd_set_digit_rule_numerative_class2e").data();
   const auto *njd_set_digit_rule_numerative_class2f = rules.table("njd_set_digit_rule_numerative_class2f").data();
   const auto *njd_set_digit_rule_numerative_class3 = rules.table("njd_set_digit_rule_numerative_class3").data();
   int i, j;
   Node *s = nullptr;
   Node *e = nullptr;
   Node *node;
   int find = 0;
   for (node = njd->head; node != nullptr; node = node->next) {
      if (find == 0 && strcmp(get(node->pos_group1), NJD_SET_DIGIT_KAZU) == 0)
         find = 1;
      if (get_digit(node, 1) >= 0 ||
          (strcmp(get(node->pos_group1), NJD_SET_DIGIT_KAZU) == 0 &&
           (is_period(get(node->string)) == 1 || is_comma(get(node->string)) == 1)
          )) {
         if (s == nullptr)
            s = node;
         if (node == njd->tail)
            e = node;
      } else {
         if (s != nullptr)
            e = node->prev;
      }
      if (s != nullptr && e != nullptr) {
         convert_digit_sequence(njd, s, e);
         s = e = nullptr;
      }
   }
   if (find == 0)
      return;
   njd->remove_silent();
   if (njd->head == nullptr)
      return;
   for (node = njd->head->next; node != nullptr && node->next != nullptr;) {
      if (strcmp(get(node->string), "*") != 0
          && strcmp(get(node->prev->string), "*") != 0
          && is_period(get(node->string)) == 1
          && strcmp(get(node->prev->pos_group1), NJD_SET_DIGIT_KAZU) == 0
          && strcmp(get(node->next->pos_group1), NJD_SET_DIGIT_KAZU) == 0) {
         document.load(node, NJD_SET_DIGIT_TEN_FEATURE);
         node->chain_flag = 1;
         if (strcmp(get(node->prev->string), NJD_SET_DIGIT_ZERO1) == 0
             || strcmp(get(node->prev->string), NJD_SET_DIGIT_ZERO2) == 0) {
            assign(node->prev->pron, NJD_SET_DIGIT_ZERO_BEFORE_DP);
            node->prev->mora_size = 2;
         } else if (strcmp(get(node->prev->string), NJD_SET_DIGIT_TWO) == 0) {
            assign(node->prev->pron, NJD_SET_DIGIT_TWO_BEFORE_DP);
            node->prev->mora_size = 2;
         } else if (strcmp(get(node->prev->string), NJD_SET_DIGIT_FIVE) == 0) {
            assign(node->prev->pron, NJD_SET_DIGIT_FIVE_BEFORE_DP);
            node->prev->mora_size = 2;
         } else if (strcmp(get(node->prev->string), NJD_SET_DIGIT_SIX) == 0) {
            node->prev->acc = 1;
         }
         node = node->next;
         while (node && strcmp(get(node->pos), NJD_SET_DIGIT_MEISHI) == 0)
            node = node->next;
         if (node)
            node = node->next;
      } else {
         node = node->next;
      }
   }
   for (node = njd->head->next; node != nullptr; node = node->next) {
      if (strcmp(get(node->prev->pos_group1), NJD_SET_DIGIT_KAZU) == 0) {
         if (strcmp(get(node->pos_group2), NJD_SET_DIGIT_JOSUUSHI) == 0
             || strcmp(get(node->pos_group1), NJD_SET_DIGIT_FUKUSHIKANOU) == 0) {
            if (search_numerative_class(njd_set_digit_rule_numerative_class1b, node) == 1)
               convert_digit_pron(njd_set_digit_rule_conv_table1b, node->prev);
            else if (search_numerative_class(njd_set_digit_rule_numerative_class1c1, node) == 1)
               convert_digit_pron(njd_set_digit_rule_conv_table1c1, node->prev);
            else if (search_numerative_class(njd_set_digit_rule_numerative_class1c2, node) == 1)
               convert_digit_pron(njd_set_digit_rule_conv_table1c2, node->prev);
            else if (search_numerative_class(njd_set_digit_rule_numerative_class1d, node) == 1)
               convert_digit_pron(njd_set_digit_rule_conv_table1d, node->prev);
            else if (search_numerative_class(njd_set_digit_rule_numerative_class1e, node) == 1)
               convert_digit_pron(njd_set_digit_rule_conv_table1e, node->prev);
            else if (search_numerative_class(njd_set_digit_rule_numerative_class1f, node) == 1)
               convert_digit_pron(njd_set_digit_rule_conv_table1f, node->prev);
            else if (search_numerative_class(njd_set_digit_rule_numerative_class1g, node) == 1)
               convert_digit_pron(njd_set_digit_rule_conv_table1g, node->prev);
            else if (search_numerative_class(njd_set_digit_rule_numerative_class1h, node) == 1)
               convert_digit_pron(njd_set_digit_rule_conv_table1h, node->prev);
            else if (search_numerative_class(njd_set_digit_rule_numerative_class1i, node) == 1)
               convert_digit_pron(njd_set_digit_rule_conv_table1i, node->prev);
            else if (search_numerative_class(njd_set_digit_rule_numerative_class1j, node) == 1)
               convert_digit_pron(njd_set_digit_rule_conv_table1j, node->prev);
            else if (search_numerative_class(njd_set_digit_rule_numerative_class1k, node) == 1)
               convert_digit_pron(njd_set_digit_rule_conv_table1k, node->prev);
            if (search_numerative_class(njd_set_digit_rule_numerative_class2b, node) == 1)
               convert_numerative_pron(njd_set_digit_rule_conv_table2b, node->prev, node);
            else if (search_numerative_class(njd_set_digit_rule_numerative_class2c, node) == 1)
               convert_numerative_pron(njd_set_digit_rule_conv_table2c, node->prev, node);
            else if (search_numerative_class(njd_set_digit_rule_numerative_class2d, node) == 1)
               convert_numerative_pron(njd_set_digit_rule_conv_table2d, node->prev, node);
            else if (search_numerative_class(njd_set_digit_rule_numerative_class2e, node) == 1)
               convert_numerative_pron(njd_set_digit_rule_conv_table2e, node->prev, node);
            else if (search_numerative_class(njd_set_digit_rule_numerative_class2f, node) == 1)
               convert_numerative_pron(njd_set_digit_rule_conv_table2f, node->prev, node);
            node->prev->chain_flag = 0;
            node->chain_flag = 1;
         }
      }
   }
   for (node = njd->head->next; node != nullptr; node = node->next) {
      if (strcmp(get(node->prev->pos_group1), NJD_SET_DIGIT_KAZU) == 0) {
         if (strcmp(get(node->pos_group1), NJD_SET_DIGIT_KAZU) == 0
             && get(node->string) != nullptr) {
            find = 0;
            for (i = 0; njd_set_digit_rule_numeral_list4[i] != nullptr; i++) {
               if (strcmp(get(node->prev->string), njd_set_digit_rule_numeral_list4[i]) == 0) {
                  for (j = 0; njd_set_digit_rule_numeral_list5[j] != nullptr; j++) {
                     if (strcmp(get(node->string), njd_set_digit_rule_numeral_list5[j]) == 0) {
                        node->prev->chain_flag = 0;
                        node->chain_flag = 1;
                        find = 1;
                        break;
                     }
                  }
                  break;
               }
            }
            if (find == 0) {
               for (i = 0; njd_set_digit_rule_numeral_list5[i] != nullptr; i++) {
                  if (strcmp(get(node->prev->string), njd_set_digit_rule_numeral_list5[i]) ==
                      0) {
                     for (j = 0; njd_set_digit_rule_numeral_list4[j] != nullptr; j++) {
                        if (strcmp(get(node->string), njd_set_digit_rule_numeral_list4[j]) ==
                            0) {
                           node->chain_flag = 0;
                           break;
                        }
                     }
                     break;
                  }
               }
            }
         }
         if (search_numerative_class(njd_set_digit_rule_numeral_list8, node) == 1)
            convert_digit_pron(njd_set_digit_rule_numeral_list9, node->prev);
         if (search_numerative_class(njd_set_digit_rule_numeral_list10, node) == 1)
            convert_digit_pron(njd_set_digit_rule_numeral_list11, node->prev);
         if (search_numerative_class(njd_set_digit_rule_numeral_list6, node) == 1)
            convert_numerative_pron(njd_set_digit_rule_numeral_list7, node->prev, node);
      }
   }
   for (node = njd->head; node != nullptr; node = node->next) {
      if (node->next != nullptr &&
          strcmp(get(node->next->string), "*") != 0 &&
          strcmp(get(node->pos_group1), NJD_SET_DIGIT_KAZU) == 0 &&
          (node->prev == nullptr
           || strcmp(get(node->prev->pos), NJD_SET_DIGIT_KIGOU) == 0
           || strcmp(get(node->prev->pos_group1), NJD_SET_DIGIT_KAZU) != 0)
          && (strcmp(get(node->next->pos_group2), NJD_SET_DIGIT_JOSUUSHI) == 0
              || strcmp(get(node->next->pos_group1), NJD_SET_DIGIT_FUKUSHIKANOU) == 0)) {
         for (i = 0; njd_set_digit_rule_numerative_class3[i] != nullptr; i += 2) {
            if (strcmp(get(node->next->string), njd_set_digit_rule_numerative_class3[i]) == 0
                && strcmp(get(node->next->read),
                          njd_set_digit_rule_numerative_class3[i + 1]) == 0) {
               for (j = 0; njd_set_digit_rule_conv_table3[j] != nullptr; j += 4) {
                  if (strcmp(get(node->string), njd_set_digit_rule_conv_table3[j]) == 0) {
                     assign(node->read, njd_set_digit_rule_conv_table3[j + 1]);
                     assign(node->pron, njd_set_digit_rule_conv_table3[j + 1]);
                     node->acc = std::max(0, atoi(njd_set_digit_rule_conv_table3[j + 2]));
                     node->mora_size = std::max(0, atoi(njd_set_digit_rule_conv_table3[j + 3]));
                     break;
                  }
               }
               break;
            }
         }
         if (strcmp(get(node->next->string), NJD_SET_DIGIT_NIN) == 0) {
            for (i = 0; njd_set_digit_rule_conv_table4[i] != nullptr; i += 2) {
               if (strcmp(get(node->string), njd_set_digit_rule_conv_table4[i]) == 0) {
                  document.load(node, njd_set_digit_rule_conv_table4[i + 1]);
                  assign(node->next->pron, nullptr);
                  break;
               }
            }
         }
         if (strcmp(get(node->next->string), NJD_SET_DIGIT_NICHI) == 0
             && strcmp(get(node->string), "*") != 0) {
            if (node->prev != nullptr
                && strstr(get(node->prev->string), NJD_SET_DIGIT_GATSU) != nullptr
                && strcmp(get(node->string), NJD_SET_DIGIT_ONE) == 0) {
               document.load(node, NJD_SET_DIGIT_TSUITACHI);
               assign(node->next->pron, nullptr);
            } else {
               for (i = 0; njd_set_digit_rule_conv_table5[i] != nullptr; i += 2) {
                  if (strcmp(get(node->string), njd_set_digit_rule_conv_table5[i]) == 0) {
                     document.load(node, njd_set_digit_rule_conv_table5[i + 1]);
                     assign(node->next->pron, nullptr);
                     break;
                  }
               }
            }
         } else if (strcmp(get(node->next->string), NJD_SET_DIGIT_NICHIKAN) == 0) {
            for (i = 0; njd_set_digit_rule_conv_table6[i] != nullptr; i += 2) {
               if (strcmp(get(node->string), njd_set_digit_rule_conv_table6[i]) == 0) {
                  document.load(node, njd_set_digit_rule_conv_table6[i + 1]);
                  assign(node->next->pron, nullptr);
                  break;
               }
            }
         }
      }
   }
   for (node = njd->head; node != nullptr; node = node->next) {
      if ((node->prev == nullptr
           || strcmp(get(node->prev->pos_group1), NJD_SET_DIGIT_KAZU) != 0)
          && node->next != nullptr && node->next->next != nullptr) {
         if (strcmp(get(node->string), NJD_SET_DIGIT_TEN) == 0
             && strcmp(get(node->next->string), NJD_SET_DIGIT_FOUR) == 0) {
            if (strcmp(get(node->next->next->string), NJD_SET_DIGIT_NICHI) == 0) {
               document.load(node, NJD_SET_DIGIT_JUYOKKA);
               assign(node->next->pron, nullptr);
               assign(node->next->next->pron, nullptr);
            } else if (strcmp(get(node->next->next->string), NJD_SET_DIGIT_NICHIKAN) == 0) {
               document.load(node, NJD_SET_DIGIT_JUYOKKAKAN);
               assign(node->next->pron, nullptr);
               assign(node->next->next->pron, nullptr);
            }
         } else if (strcmp(get(node->string), NJD_SET_DIGIT_TWO) == 0
                    && strcmp(get(node->next->string), NJD_SET_DIGIT_TEN) == 0) {
            if (strcmp(get(node->next->next->string), NJD_SET_DIGIT_NICHI) == 0) {
               document.load(node, NJD_SET_DITIT_HATSUKA);
               assign(node->next->pron, nullptr);
               assign(node->next->next->pron, nullptr);
            } else if (strcmp(get(node->next->next->string), NJD_SET_DIGIT_NICHIKAN) == 0) {
               document.load(node, NJD_SET_DIGIT_HATSUKAKAN);
               assign(node->next->pron, nullptr);
               assign(node->next->next->pron, nullptr);
            } else if (strcmp(get(node->next->next->string), NJD_SET_DIGIT_FOUR) == 0
                       && node->next->next->next != nullptr) {
               if (strcmp(get(node->next->next->next->string), NJD_SET_DIGIT_NICHI) == 0) {
                  document.load(node, NJD_SET_DIGIT_NIJU);
                  document.load(node->next, NJD_SET_DITIT_YOKKA);
                  assign(node->next->next->pron, nullptr);
                  assign(node->next->next->next->pron, nullptr);
               } else if (strcmp(get(node->next->next->next->string), NJD_SET_DIGIT_NICHIKAN)
                          == 0) {
                  document.load(node, NJD_SET_DIGIT_NIJU);
                  document.load(node->next, NJD_SET_DIGIT_YOKKAKAN);
                  assign(node->next->next->pron, nullptr);
                  assign(node->next->next->next->pron, nullptr);
               }
            }
         }
      }
   }
   njd->remove_silent();
   if (njd->head == nullptr)
      return;
}


// Ordered grammar constraints: later rules can override earlier chain decisions.
void set_accent_phrase(Document * njd)
{
   const auto *NJD_SET_ACCENT_PHRASE_DE = rules.constant("NJD_SET_ACCENT_PHRASE_DE");
   const auto *NJD_SET_ACCENT_PHRASE_DOUSHI = rules.constant("NJD_SET_ACCENT_PHRASE_DOUSHI");
   const auto *NJD_SET_ACCENT_PHRASE_FUKUSHI = rules.constant("NJD_SET_ACCENT_PHRASE_FUKUSHI");
   const auto *NJD_SET_ACCENT_PHRASE_FUKUSHI_KANOU = rules.constant("NJD_SET_ACCENT_PHRASE_FUKUSHI_KANOU");
   const auto *NJD_SET_ACCENT_PHRASE_HIJIRITSU = rules.constant("NJD_SET_ACCENT_PHRASE_HIJIRITSU");
   const auto *NJD_SET_ACCENT_PHRASE_JODOUSHI = rules.constant("NJD_SET_ACCENT_PHRASE_JODOUSHI");
   const auto *NJD_SET_ACCENT_PHRASE_JOSHI = rules.constant("NJD_SET_ACCENT_PHRASE_JOSHI");
   const auto *NJD_SET_ACCENT_PHRASE_KEIYOUDOUSHI_GOKAN = rules.constant("NJD_SET_ACCENT_PHRASE_KEIYOUDOUSHI_GOKAN");
   const auto *NJD_SET_ACCENT_PHRASE_KEIYOUSHI = rules.constant("NJD_SET_ACCENT_PHRASE_KEIYOUSHI");
   const auto *NJD_SET_ACCENT_PHRASE_KIGOU = rules.constant("NJD_SET_ACCENT_PHRASE_KIGOU");
   const auto *NJD_SET_ACCENT_PHRASE_MEI = rules.constant("NJD_SET_ACCENT_PHRASE_MEI");
   const auto *NJD_SET_ACCENT_PHRASE_MEISHI = rules.constant("NJD_SET_ACCENT_PHRASE_MEISHI");
   const auto *NJD_SET_ACCENT_PHRASE_RENTAISHI = rules.constant("NJD_SET_ACCENT_PHRASE_RENTAISHI");
   const auto *NJD_SET_ACCENT_PHRASE_RENYOU = rules.constant("NJD_SET_ACCENT_PHRASE_RENYOU");
   const auto *NJD_SET_ACCENT_PHRASE_SAHEN_SETSUZOKU = rules.constant("NJD_SET_ACCENT_PHRASE_SAHEN_SETSUZOKU");
   const auto *NJD_SET_ACCENT_PHRASE_SEI = rules.constant("NJD_SET_ACCENT_PHRASE_SEI");
   const auto *NJD_SET_ACCENT_PHRASE_SETSUBI = rules.constant("NJD_SET_ACCENT_PHRASE_SETSUBI");
   const auto *NJD_SET_ACCENT_PHRASE_SETSUZOKUJOSHI = rules.constant("NJD_SET_ACCENT_PHRASE_SETSUZOKUJOSHI");
   const auto *NJD_SET_ACCENT_PHRASE_SETSUZOKUSHI = rules.constant("NJD_SET_ACCENT_PHRASE_SETSUZOKUSHI");
   const auto *NJD_SET_ACCENT_PHRASE_SETTOUSHI = rules.constant("NJD_SET_ACCENT_PHRASE_SETTOUSHI");
   const auto *NJD_SET_ACCENT_PHRASE_TE = rules.constant("NJD_SET_ACCENT_PHRASE_TE");
   Node *node;
   if (njd == nullptr || njd->head == nullptr)
      return;
   for (node = njd->head->next; node != nullptr; node = node->next) {
      if (node->chain_flag < 0) {
         node->chain_flag = 1;
         if (strcmp(get(node->prev->pos), NJD_SET_ACCENT_PHRASE_MEISHI) == 0)
            if (strcmp(get(node->pos), NJD_SET_ACCENT_PHRASE_MEISHI) == 0)
               node->chain_flag = 1;
         if (strcmp(get(node->prev->pos), NJD_SET_ACCENT_PHRASE_KEIYOUSHI) == 0)
            if (strcmp(get(node->pos), NJD_SET_ACCENT_PHRASE_MEISHI) == 0)
               node->chain_flag = 0;
         if (strcmp(get(node->prev->pos), NJD_SET_ACCENT_PHRASE_MEISHI) == 0)
            if (strcmp
                (get(node->prev->pos_group1), NJD_SET_ACCENT_PHRASE_KEIYOUDOUSHI_GOKAN) == 0)
               if (strcmp(get(node->pos), NJD_SET_ACCENT_PHRASE_MEISHI) == 0)
                  node->chain_flag = 0;
         if (strcmp(get(node->prev->pos), NJD_SET_ACCENT_PHRASE_DOUSHI) == 0) {
            if (strcmp(get(node->pos), NJD_SET_ACCENT_PHRASE_KEIYOUSHI) == 0)
               node->chain_flag = 0;
            else if (strcmp(get(node->pos), NJD_SET_ACCENT_PHRASE_MEISHI) == 0)
               node->chain_flag = 0;
         }
         if (strcmp(get(node->pos), NJD_SET_ACCENT_PHRASE_FUKUSHI) == 0
             || strcmp(get(node->prev->pos), NJD_SET_ACCENT_PHRASE_FUKUSHI) == 0
             || strcmp(get(node->pos), NJD_SET_ACCENT_PHRASE_SETSUZOKUSHI) == 0
             || strcmp(get(node->prev->pos), NJD_SET_ACCENT_PHRASE_SETSUZOKUSHI) == 0
             || strcmp(get(node->pos), NJD_SET_ACCENT_PHRASE_RENTAISHI) == 0
             || strcmp(get(node->prev->pos), NJD_SET_ACCENT_PHRASE_RENTAISHI) == 0)
            node->chain_flag = 0;
         if (strcmp(get(node->prev->pos), NJD_SET_ACCENT_PHRASE_MEISHI) == 0)
            if (strcmp(get(node->prev->pos_group1), NJD_SET_ACCENT_PHRASE_FUKUSHI_KANOU)
                == 0)
               node->chain_flag = 0;
         if (strcmp(get(node->pos), NJD_SET_ACCENT_PHRASE_MEISHI) == 0)
            if (strcmp(get(node->pos_group1), NJD_SET_ACCENT_PHRASE_FUKUSHI_KANOU) == 0)
               node->chain_flag = 0;
         if (strcmp(get(node->pos), NJD_SET_ACCENT_PHRASE_JODOUSHI) == 0)
            node->chain_flag = 1;
         if (strcmp(get(node->pos), NJD_SET_ACCENT_PHRASE_JOSHI) == 0)
            node->chain_flag = 1;
         if (strcmp(get(node->prev->pos), NJD_SET_ACCENT_PHRASE_JODOUSHI) == 0)
            if (strcmp(get(node->pos), NJD_SET_ACCENT_PHRASE_JODOUSHI) != 0 &&
                strcmp(get(node->pos), NJD_SET_ACCENT_PHRASE_JOSHI) != 0)
               node->chain_flag = 0;
         if (strcmp(get(node->prev->pos), NJD_SET_ACCENT_PHRASE_JOSHI) == 0)
            if (strcmp(get(node->pos), NJD_SET_ACCENT_PHRASE_JODOUSHI) != 0 &&
                strcmp(get(node->pos), NJD_SET_ACCENT_PHRASE_JOSHI) != 0)
               node->chain_flag = 0;
         if (strcmp(get(node->prev->pos_group1), NJD_SET_ACCENT_PHRASE_SETSUBI) == 0)
            if (strcmp(get(node->pos), NJD_SET_ACCENT_PHRASE_MEISHI) == 0)
               node->chain_flag = 0;
         if (strcmp(get(node->pos), NJD_SET_ACCENT_PHRASE_KEIYOUSHI) == 0)
            if (strcmp(get(node->pos_group1), NJD_SET_ACCENT_PHRASE_HIJIRITSU) == 0) {
               if (strcmp(get(node->prev->pos), NJD_SET_ACCENT_PHRASE_DOUSHI) == 0) {
                  if (strtopcmp(get(node->prev->cform), NJD_SET_ACCENT_PHRASE_RENYOU) != -1)
                     node->chain_flag = 1;
               } else if (strcmp(get(node->prev->pos), NJD_SET_ACCENT_PHRASE_KEIYOUSHI)
                          == 0) {
                  if (strtopcmp(get(node->prev->cform), NJD_SET_ACCENT_PHRASE_RENYOU) != -1)
                     node->chain_flag = 1;
               } else if (strcmp(get(node->prev->pos), NJD_SET_ACCENT_PHRASE_JOSHI) == 0) {
                  if (strcmp
                      (get(node->prev->pos_group1),
                       NJD_SET_ACCENT_PHRASE_SETSUZOKUJOSHI) == 0) {
                     if (strcmp(get(node->prev->string), NJD_SET_ACCENT_PHRASE_TE) == 0)
                        node->chain_flag = 1;
                     else if (strcmp(get(node->prev->string), NJD_SET_ACCENT_PHRASE_DE) == 0)
                        node->chain_flag = 1;
                  }
               }
            }
         if (strcmp(get(node->pos), NJD_SET_ACCENT_PHRASE_DOUSHI) == 0)
            if (strcmp(get(node->pos_group1), NJD_SET_ACCENT_PHRASE_HIJIRITSU) == 0) {
               if (strcmp(get(node->prev->pos), NJD_SET_ACCENT_PHRASE_DOUSHI) == 0) {
                  if (strtopcmp(get(node->prev->cform), NJD_SET_ACCENT_PHRASE_RENYOU) != -1)
                     node->chain_flag = 1;
               } else if (strcmp(get(node->prev->pos), NJD_SET_ACCENT_PHRASE_MEISHI) == 0) {
                  if (strcmp
                      (get(node->prev->pos_group1),
                       NJD_SET_ACCENT_PHRASE_SAHEN_SETSUZOKU) == 0)
                     node->chain_flag = 1;
               }
            }
         if (strcmp(get(node->prev->pos), NJD_SET_ACCENT_PHRASE_MEISHI) == 0) {
            if (strcmp(get(node->pos), NJD_SET_ACCENT_PHRASE_DOUSHI) == 0 ||
                strcmp(get(node->pos), NJD_SET_ACCENT_PHRASE_KEIYOUSHI) == 0 ||
                strcmp(get(node->pos_group1), NJD_SET_ACCENT_PHRASE_KEIYOUDOUSHI_GOKAN) == 0)
               node->chain_flag = 0;
         }
         if (strcmp(get(node->pos), NJD_SET_ACCENT_PHRASE_KIGOU) == 0 ||
             strcmp(get(node->prev->pos), NJD_SET_ACCENT_PHRASE_KIGOU) == 0)
            node->chain_flag = 0;
         if (strcmp(get(node->pos), NJD_SET_ACCENT_PHRASE_SETTOUSHI) == 0)
            node->chain_flag = 0;
         if (strcmp(get(node->prev->pos_group3), NJD_SET_ACCENT_PHRASE_SEI) == 0
             && strcmp(get(node->pos), NJD_SET_ACCENT_PHRASE_MEISHI) == 0)
            node->chain_flag = 0;
         if (strcmp(get(node->prev->pos), NJD_SET_ACCENT_PHRASE_MEISHI) == 0
             && strcmp(get(node->pos_group3), NJD_SET_ACCENT_PHRASE_MEI) == 0)
            node->chain_flag = 0;
         if (strcmp(get(node->pos_group1), NJD_SET_ACCENT_PHRASE_SETSUBI) == 0)
            node->chain_flag = 1;
      }
   }
}

void set_accent_type(Document * njd)
{
   const auto *NJD_SET_ACCENT_TYPE_CHOU = rules.constant("NJD_SET_ACCENT_TYPE_CHOU");
   const auto *NJD_SET_ACCENT_TYPE_GO = rules.constant("NJD_SET_ACCENT_TYPE_GO");
   const auto *NJD_SET_ACCENT_TYPE_HACHI = rules.constant("NJD_SET_ACCENT_TYPE_HACHI");
   const auto *NJD_SET_ACCENT_TYPE_HYAKU = rules.constant("NJD_SET_ACCENT_TYPE_HYAKU");
   const auto *NJD_SET_ACCENT_TYPE_ICHI = rules.constant("NJD_SET_ACCENT_TYPE_ICHI");
   const auto *NJD_SET_ACCENT_TYPE_IKU = rules.constant("NJD_SET_ACCENT_TYPE_IKU");
   const auto *NJD_SET_ACCENT_TYPE_JYUU = rules.constant("NJD_SET_ACCENT_TYPE_JYUU");
   const auto *NJD_SET_ACCENT_TYPE_KAZU = rules.constant("NJD_SET_ACCENT_TYPE_KAZU");
   const auto *NJD_SET_ACCENT_TYPE_KYUU = rules.constant("NJD_SET_ACCENT_TYPE_KYUU");
   const auto *NJD_SET_ACCENT_TYPE_MAN = rules.constant("NJD_SET_ACCENT_TYPE_MAN");
   const auto *NJD_SET_ACCENT_TYPE_NAN = rules.constant("NJD_SET_ACCENT_TYPE_NAN");
   const auto *NJD_SET_ACCENT_TYPE_NANA = rules.constant("NJD_SET_ACCENT_TYPE_NANA");
   const auto *NJD_SET_ACCENT_TYPE_NI = rules.constant("NJD_SET_ACCENT_TYPE_NI");
   const auto *NJD_SET_ACCENT_TYPE_OKU = rules.constant("NJD_SET_ACCENT_TYPE_OKU");
   const auto *NJD_SET_ACCENT_TYPE_ROKU = rules.constant("NJD_SET_ACCENT_TYPE_ROKU");
   const auto *NJD_SET_ACCENT_TYPE_SAN = rules.constant("NJD_SET_ACCENT_TYPE_SAN");
   const auto *NJD_SET_ACCENT_TYPE_SEN = rules.constant("NJD_SET_ACCENT_TYPE_SEN");
   const auto *NJD_SET_ACCENT_TYPE_YON = rules.constant("NJD_SET_ACCENT_TYPE_YON");
   Node *node;
   Node *top_node = nullptr;
   std::string rule;
   int add_type = 0;
   int mora_size = 0;
   if (njd == nullptr || njd->head == nullptr)
      return;
   for (node = njd->head; node != nullptr; node = node->next) {
      if ((node == njd->head) || (node->chain_flag != 1)) {
         top_node = node;
         mora_size = 0;
      } else if (node->prev != nullptr && node->chain_flag == 1) {
         rule = accent_rule(get(node->chain_rule), get(node->prev->pos), add_type);
         if (rule == "F2") {
            if (top_node->acc == 0)
               top_node->acc = std::max(0, mora_size + add_type);
         } else if (rule == "F3") {
            if (top_node->acc != 0)
               top_node->acc = std::max(0, mora_size + add_type);
         } else if (rule == "F4") {
            top_node->acc = std::max(0, mora_size + add_type);
         } else if (rule == "F5") {
            top_node->acc = 0;
         } else if (rule == "C1") {
            top_node->acc = std::max(0, mora_size + node->acc);
         } else if (rule == "C2") {
            top_node->acc = std::max(0, mora_size + 1);
         } else if (rule == "C3") {
            top_node->acc = std::max(0, mora_size);
         } else if (rule == "C4") {
            top_node->acc = 0;
         } else if (rule == "P1") {
            if (node->acc == 0)
               top_node->acc = 0;
            else
               top_node->acc = std::max(0, mora_size + node->acc);
         } else if (rule == "P2") {
            if (node->acc == 0)
               top_node->acc = std::max(0, mora_size + 1);
            else
               top_node->acc = std::max(0, mora_size + node->acc);
         } else if (rule == "P6") {
            top_node->acc = 0;
         } else if (rule == "P14") {
            if (node->acc != 0)
               top_node->acc = std::max(0, mora_size + node->acc);
         }
      }
      if (node->prev != nullptr && node->chain_flag == 1 &&
          strcmp(get(node->prev->pos_group1), NJD_SET_ACCENT_TYPE_KAZU) == 0 &&
          strcmp(get(node->pos_group1), NJD_SET_ACCENT_TYPE_KAZU) == 0) {
         if (strcmp(get(node->string), NJD_SET_ACCENT_TYPE_JYUU) == 0) {
            node->prev->acc = 1;
            if ((strcmp(get(node->prev->string), NJD_SET_ACCENT_TYPE_GO) == 0 ||
                 strcmp(get(node->prev->string), NJD_SET_ACCENT_TYPE_ROKU) == 0 ||
                 strcmp(get(node->prev->string), NJD_SET_ACCENT_TYPE_HACHI) == 0)) {
               if (node->next != nullptr && (strcmp(get(node->next->string), NJD_SET_ACCENT_TYPE_ICHI) == 0
                       || strcmp(get(node->next->string), NJD_SET_ACCENT_TYPE_NI) == 0
                       || strcmp(get(node->next->string), NJD_SET_ACCENT_TYPE_SAN) == 0
                       || strcmp(get(node->next->string), NJD_SET_ACCENT_TYPE_YON) == 0
                       || strcmp(get(node->next->string), NJD_SET_ACCENT_TYPE_GO) == 0
                       || strcmp(get(node->next->string), NJD_SET_ACCENT_TYPE_ROKU) == 0
                       || strcmp(get(node->next->string), NJD_SET_ACCENT_TYPE_NANA) == 0
                       || strcmp(get(node->next->string), NJD_SET_ACCENT_TYPE_HACHI) == 0
                       || strcmp(get(node->next->string), NJD_SET_ACCENT_TYPE_KYUU) == 0))
                  node->prev->acc = 0;
            }
         } else if (strcmp(get(node->string), NJD_SET_ACCENT_TYPE_HYAKU) == 0) {
            if (strcmp(get(node->prev->string), NJD_SET_ACCENT_TYPE_NANA) == 0) {
               node->prev->acc = 2;
            } else if ((strcmp(get(node->prev->string), NJD_SET_ACCENT_TYPE_SAN) == 0 ||
                        strcmp(get(node->prev->string), NJD_SET_ACCENT_TYPE_YON) == 0 ||
                        strcmp(get(node->prev->string), NJD_SET_ACCENT_TYPE_KYUU) == 0 ||
                        strcmp(get(node->prev->string), NJD_SET_ACCENT_TYPE_NAN) == 0)) {
               node->prev->acc = 1;
            } else {
               node->prev->acc = std::max(0, node->prev->mora_size + node->mora_size);
            }
         } else if (strcmp(get(node->string), NJD_SET_ACCENT_TYPE_SEN) == 0) {
            node->prev->acc = std::max(0, node->prev->mora_size + 1);
         } else if (strcmp(get(node->string), NJD_SET_ACCENT_TYPE_MAN) == 0) {
            node->prev->acc = std::max(0, node->prev->mora_size + 1);
         } else if (strcmp(get(node->string), NJD_SET_ACCENT_TYPE_OKU) == 0) {
            if ((strcmp(get(node->prev->string), NJD_SET_ACCENT_TYPE_ICHI) == 0 ||
                 strcmp(get(node->prev->string), NJD_SET_ACCENT_TYPE_ROKU) == 0 ||
                 strcmp(get(node->prev->string), NJD_SET_ACCENT_TYPE_NANA) == 0 ||
                 strcmp(get(node->prev->string), NJD_SET_ACCENT_TYPE_HACHI) == 0 ||
                 strcmp(get(node->prev->string), NJD_SET_ACCENT_TYPE_IKU) == 0)) {
               node->prev->acc = 2;
            } else {
               node->prev->acc = 1;
            }
         } else if (strcmp(get(node->string), NJD_SET_ACCENT_TYPE_CHOU) == 0) {
            if ((strcmp(get(node->prev->string), NJD_SET_ACCENT_TYPE_ROKU) == 0 ||
                 strcmp(get(node->prev->string), NJD_SET_ACCENT_TYPE_NANA) == 0)) {
               node->prev->acc = 2;
            } else {
               node->prev->acc = 1;
            }
         }
      }
      if (strcmp(get(node->string), NJD_SET_ACCENT_TYPE_JYUU) == 0 &&
          node->chain_flag != 1 && node->next != nullptr &&
          strcmp(get(node->next->pos_group1), NJD_SET_ACCENT_TYPE_KAZU) == 0) {
         node->acc = 0;
      }
      mora_size += node->mora_size;
   }
}


void get_mora_information(Node * node, int index, const char **mora, Node ** nlink,
                                 int *flag, int *size, int *midx, int *atype)
{
   const auto *NJD_SET_UNVOICED_VOWEL_QUESTION = rules.constant("NJD_SET_UNVOICED_VOWEL_QUESTION");
   const auto *NJD_SET_UNVOICED_VOWEL_QUOTATION = rules.constant("NJD_SET_UNVOICED_VOWEL_QUOTATION");
   const auto *NJD_SET_UNVOICED_VOWEL_TOUTEN = rules.constant("NJD_SET_UNVOICED_VOWEL_TOUTEN");
   const auto *njd_set_unvoiced_vowel_mora_list = rules.table("njd_set_unvoiced_vowel_mora_list").data();
   int i;
   int matched_size;
   const char *str = get(node->pron);
   int len = strlen(str);
   while (index >= len && node->next) {
      index -= len;
      node = node->next;
      str = get(node->pron);
      len = strlen(str);
   }
   if (index >= len) {
      *mora = nullptr; *nlink = nullptr;
      *flag = -1; *size = *midx = *atype = 0;
      return;
   }
   *nlink = node;
   if (index == 0 && node->chain_flag != 1) {
      *midx = 0;
      *atype = node->acc;
   }
   if (strcmp(str, NJD_SET_UNVOICED_VOWEL_TOUTEN) == 0) {
      *mora = NJD_SET_UNVOICED_VOWEL_TOUTEN;
      *flag = 0;
      *size = strlen(NJD_SET_UNVOICED_VOWEL_TOUTEN);
      return;
   }
   if (strcmp(str, NJD_SET_UNVOICED_VOWEL_QUESTION) == 0) {
      *mora = NJD_SET_UNVOICED_VOWEL_QUESTION;
      *flag = 0;
      *size = strlen(NJD_SET_UNVOICED_VOWEL_QUESTION);
      return;
   }
   *mora = nullptr;
   *flag = -1;
   *size = 0;
   for (i = 0; njd_set_unvoiced_vowel_mora_list[i] != nullptr; i++) {
      matched_size = strtopcmp(&str[index], njd_set_unvoiced_vowel_mora_list[i]);
      if (matched_size > 0) {
         *mora = njd_set_unvoiced_vowel_mora_list[i];
         *size = matched_size;
         break;
      }
   }
   matched_size = strtopcmp(&str[index + (*size)], NJD_SET_UNVOICED_VOWEL_QUOTATION);
   if (matched_size > 0) {
      *flag = 1;
      *size += matched_size;
   }
}
int apply_unvoice_rule(const char *current, const char *next)
{
   const auto *njd_set_unvoiced_vowel_candidate_list1 = rules.table("njd_set_unvoiced_vowel_candidate_list1").data();
   const auto *njd_set_unvoiced_vowel_candidate_list2 = rules.table("njd_set_unvoiced_vowel_candidate_list2").data();
   const auto *njd_set_unvoiced_vowel_candidate_list3 = rules.table("njd_set_unvoiced_vowel_candidate_list3").data();
   const auto *njd_set_unvoiced_vowel_next_mora_list1 = rules.table("njd_set_unvoiced_vowel_next_mora_list1").data();
   const auto *njd_set_unvoiced_vowel_next_mora_list2 = rules.table("njd_set_unvoiced_vowel_next_mora_list2").data();
   const auto *njd_set_unvoiced_vowel_next_mora_list3 = rules.table("njd_set_unvoiced_vowel_next_mora_list3").data();
   int i, j;
   if (next == nullptr)
      return 0;
   for (i = 0; njd_set_unvoiced_vowel_candidate_list1[i] != nullptr; i++) {
      if (strcmp(current, njd_set_unvoiced_vowel_candidate_list1[i]) == 0) {
         for (j = 0; njd_set_unvoiced_vowel_next_mora_list1[j] != nullptr; j++)
            if (strtopcmp(next, njd_set_unvoiced_vowel_next_mora_list1[j]) > 0)
               return 1;
         return 0;
      }
   }
   for (i = 0; njd_set_unvoiced_vowel_candidate_list2[i] != nullptr; i++) {
      if (strcmp(current, njd_set_unvoiced_vowel_candidate_list2[i]) == 0) {
         for (j = 0; njd_set_unvoiced_vowel_next_mora_list2[j] != nullptr; j++)
            if (strtopcmp(next, njd_set_unvoiced_vowel_next_mora_list2[j]) > 0)
               return 1;
         return 0;
      }
   }
   for (i = 0; njd_set_unvoiced_vowel_candidate_list3[i] != nullptr; i++) {
      if (strcmp(current, njd_set_unvoiced_vowel_candidate_list3[i]) == 0) {
         for (j = 0; njd_set_unvoiced_vowel_next_mora_list3[j] != nullptr; j++) {
            if (strtopcmp(next, njd_set_unvoiced_vowel_next_mora_list3[j]) > 0)
               return 1;
         }
         return 0;
      }
   }
   return -1;
}
void set_unvoiced_vowel(Document * njd)
{
   const auto *NJD_SET_UNVOICED_VOWEL_CHOUON = rules.constant("NJD_SET_UNVOICED_VOWEL_CHOUON");
   const auto *NJD_SET_UNVOICED_VOWEL_DE = rules.constant("NJD_SET_UNVOICED_VOWEL_DE");
   const auto *NJD_SET_UNVOICED_VOWEL_DOUSHI = rules.constant("NJD_SET_UNVOICED_VOWEL_DOUSHI");
   const auto *NJD_SET_UNVOICED_VOWEL_FILLER = rules.constant("NJD_SET_UNVOICED_VOWEL_FILLER");
   const auto *NJD_SET_UNVOICED_VOWEL_JODOUSHI = rules.constant("NJD_SET_UNVOICED_VOWEL_JODOUSHI");
   const auto *NJD_SET_UNVOICED_VOWEL_JOSHI = rules.constant("NJD_SET_UNVOICED_VOWEL_JOSHI");
   const auto *NJD_SET_UNVOICED_VOWEL_KANDOUSHI = rules.constant("NJD_SET_UNVOICED_VOWEL_KANDOUSHI");
   const auto *NJD_SET_UNVOICED_VOWEL_MA = rules.constant("NJD_SET_UNVOICED_VOWEL_MA");
   const auto *NJD_SET_UNVOICED_VOWEL_QUESTION = rules.constant("NJD_SET_UNVOICED_VOWEL_QUESTION");
   const auto *NJD_SET_UNVOICED_VOWEL_QUOTATION = rules.constant("NJD_SET_UNVOICED_VOWEL_QUOTATION");
   const auto *NJD_SET_UNVOICED_VOWEL_SHI = rules.constant("NJD_SET_UNVOICED_VOWEL_SHI");
   const auto *NJD_SET_UNVOICED_VOWEL_SU = rules.constant("NJD_SET_UNVOICED_VOWEL_SU");
   Node *node;
   int index;
   int len;
   std::string buff;
   const char *str;
   const char *mora1 = nullptr, *mora2 = nullptr, *mora3 = nullptr;
   Node *nlink1 = nullptr, *nlink2 = nullptr, *nlink3 = nullptr;
   int size1 = 0, size2 = 0, size3 = 0;
   int flag1 = -1, flag2 = -1, flag3 = -1;
   int midx1 = 0, midx2 = 1, midx3 = 2;
   int atype1 = 0, atype2 = 0, atype3 = 0;
   for (node = njd->head; node != nullptr; node = node->next) {
      buff.clear();
      str = get(node->pron);
      len = strlen(str);
      for (index = 0; index < len;) {
         if (mora1 == nullptr)
            get_mora_information(node, index, &mora1, &nlink1, &flag1, &size1, &midx1, &atype1);
         if (mora1 == nullptr) {

            return;
         }
         if (mora2 == nullptr) {
            midx2 = midx1 + 1;
            atype2 = atype1;
            get_mora_information(node, index + size1, &mora2, &nlink2, &flag2, &size2, &midx2,
                                 &atype2);
         }
         if (mora3 == nullptr) {
            midx3 = midx2 + 1;
            atype3 = atype2;
            get_mora_information(node, index + size1 + size2, &mora3, &nlink3, &flag3, &size3,
                                 &midx3, &atype3);
         }
         if (mora2 != nullptr && mora3 != nullptr && nlink1 == nlink2 && nlink2 != nlink3 &&
             (strcmp(mora1, NJD_SET_UNVOICED_VOWEL_MA) == 0
              || strcmp(mora1, NJD_SET_UNVOICED_VOWEL_DE) == 0)
             && strcmp(mora2, NJD_SET_UNVOICED_VOWEL_SU) == 0
             && (strcmp(get(nlink2->pos), NJD_SET_UNVOICED_VOWEL_DOUSHI) == 0
                 || strcmp(get(nlink2->pos), NJD_SET_UNVOICED_VOWEL_JODOUSHI) == 0
                 || strcmp(get(nlink2->pos), NJD_SET_UNVOICED_VOWEL_KANDOUSHI) == 0)
             ) {
            if (strcmp(get(nlink3->pron), NJD_SET_UNVOICED_VOWEL_QUESTION) == 0
                || strcmp(get(nlink3->pron), NJD_SET_UNVOICED_VOWEL_CHOUON) == 0)
               flag2 = 0;
            else
               flag2 = 1;
         }
         if (flag1 != 1 && flag2 == -1 && flag3 != 1 && mora2 != nullptr &&
             strcmp(get(nlink2->pron), NJD_SET_UNVOICED_VOWEL_SHI) == 0 &&
             (strcmp(get(nlink2->pos), NJD_SET_UNVOICED_VOWEL_DOUSHI) == 0 ||
              strcmp(get(nlink2->pos), NJD_SET_UNVOICED_VOWEL_JODOUSHI) == 0 ||
              strcmp(get(nlink2->pos), NJD_SET_UNVOICED_VOWEL_JOSHI) == 0)) {
            if (atype2 == midx2 + 1) {
               flag2 = 0;
            } else {
               flag2 = apply_unvoice_rule(mora2, mora3);
            }
            if (flag2 == 1) {
               if (flag1 == -1)
                  flag1 = 0;
               if (flag3 == -1)
                  flag3 = 0;
            }
         }
         if (flag1 == -1) {
            if (strcmp(get(nlink1->pos), NJD_SET_UNVOICED_VOWEL_FILLER) == 0) {
               flag1 = 0;
            } else if (flag2 == 1) {
               flag1 = 0;
            } else if (atype1 == midx1 + 1) {
               flag1 = 0;
            } else {
               flag1 = apply_unvoice_rule(mora1, mora2);
            }
         }
         if (flag1 == 1 && flag2 == -1) {
            flag2 = 0;
         }
         buff += mora1;
         if (flag1 == 1)
            buff += NJD_SET_UNVOICED_VOWEL_QUOTATION;
         index += size1;
         mora1 = mora2;
         nlink1 = nlink2;
         size1 = size2;
         flag1 = flag2;
         midx1 = midx2;
         atype1 = atype2;
         mora2 = mora3;
         nlink2 = nlink3;
         size2 = size3;
         flag2 = flag3;
         midx2 = midx3;
         atype2 = atype3;
         mora3 = nullptr;
         nlink3 = nullptr;
         size3 = 0;
         flag3 = -1;
         midx3 = 0;
         atype3 = 0;
      }
      assign(node->pron, buff.c_str());
   }
}

};
} // namespace
void pronounce(Document &document, const Rules &rules) {
    Process process{document, rules};
    process.set_pronunciation(&document);
    process.set_digit(&document);
    process.set_accent_phrase(&document);
    process.set_accent_type(&document);
    process.set_unvoiced_vowel(&document);
    // Open JTalk 1.11 disables its deprecated long-vowel estimator. Long
    // marks still become the preceding vowel in the prosody conversion.
}
} // namespace tts::text::japanese
