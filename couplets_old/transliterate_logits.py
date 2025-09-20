"""
Multi-library transliteration script for various scripts
Handles Latin detection and routes to appropriate specialized libraries
"""

import unicodedata
import re
from collections import Counter
from typing import Optional

from transliterate import translit
from pypinyin import lazy_pinyin
from gimeltra import tr as gimeltra_tr
import transliter as tl
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate as indic_translit
from pythainlp.transliterate import romanize


def is_all_latin(text):
   """
   Check if text contains only Latin characters (including diacritics).
   Returns True if all alphabetic characters are Latin.
   """
   for char in text:
       if char.isalpha():
           code_point = ord(char)
           if not (
               (0x0041 <= code_point <= 0x005A) or  # A-Z
               (0x0061 <= code_point <= 0x007A) or  # a-z
               (0x00C0 <= code_point <= 0x00FF) or  # Latin-1 Supplement
               (0x0100 <= code_point <= 0x017F) or  # Latin Extended-A
               (0x0180 <= code_point <= 0x024F) or  # Latin Extended-B
               (0x1E00 <= code_point <= 0x1EFF)     # Latin Extended Additional
           ):
               return False
   return True


def detect_primary_script(text):
   """
   Detect the primary script of the text based on character frequency.
   Returns the name of the most frequent script.
   """
   scripts = Counter()
   
   for char in text:
       if char.isalpha():
           code_point = ord(char)
           
           # Latin
           if ((0x0041 <= code_point <= 0x005A) or 
               (0x0061 <= code_point <= 0x007A) or
               (0x00C0 <= code_point <= 0x00FF) or
               (0x0100 <= code_point <= 0x017F) or
               (0x0180 <= code_point <= 0x024F) or
               (0x1E00 <= code_point <= 0x1EFF)):
               scripts['latin'] += 1
           
           # Chinese (CJK Unified Ideographs)
           elif 0x4e00 <= code_point <= 0x9fff:
               scripts['chinese'] += 1
           
           # Japanese Hiragana
           elif 0x3040 <= code_point <= 0x309f:
               scripts['hiragana'] += 1
           
           # Japanese Katakana
           elif 0x30a0 <= code_point <= 0x30ff:
               scripts['katakana'] += 1
           
           # Korean Hangul
           elif 0xac00 <= code_point <= 0xd7af:
               scripts['hangul'] += 1
           
           # Arabic
           elif 0x0600 <= code_point <= 0x06ff:
               scripts['arabic'] += 1
           
           # Hebrew
           elif 0x0590 <= code_point <= 0x05ff:
               scripts['hebrew'] += 1
           
           # Cyrillic
           elif 0x0400 <= code_point <= 0x04ff:
               scripts['cyrillic'] += 1
           
           # Armenian
           elif 0x0530 <= code_point <= 0x058f:
               scripts['armenian'] += 1
           
           # Georgian
           elif 0x10a0 <= code_point <= 0x10ff:
               scripts['georgian'] += 1
           
           # Greek
           elif 0x0370 <= code_point <= 0x03ff:
               scripts['greek'] += 1
           
           # Thai
           elif 0x0e00 <= code_point <= 0x0e7f:
               scripts['thai'] += 1
           
           # Devanagari (Hindi, Sanskrit, Marathi, Nepali)
           elif 0x0900 <= code_point <= 0x097f:
               scripts['devanagari'] += 1
           
           # Bengali
           elif 0x0980 <= code_point <= 0x09ff:
               scripts['bengali'] += 1
           
           # Gurmukhi (Punjabi)
           elif 0x0a00 <= code_point <= 0x0a7f:
               scripts['gurmukhi'] += 1
           
           # Gujarati
           elif 0x0a80 <= code_point <= 0x0aff:
               scripts['gujarati'] += 1
           
           # Oriya
           elif 0x0b00 <= code_point <= 0x0b7f:
               scripts['oriya'] += 1
           
           # Tamil
           elif 0x0b80 <= code_point <= 0x0bff:
               scripts['tamil'] += 1
           
           # Telugu
           elif 0x0c00 <= code_point <= 0x0c7f:
               scripts['telugu'] += 1
           
           # Kannada
           elif 0x0c80 <= code_point <= 0x0cff:
               scripts['kannada'] += 1
           
           # Malayalam
           elif 0x0d00 <= code_point <= 0x0d7f:
               scripts['malayalam'] += 1
           
           # Sinhala
           elif 0x0d80 <= code_point <= 0x0dff:
               scripts['sinhala'] += 1
           
           else:
               scripts['other'] += 1
   
   # Return the most frequent script, or 'unknown' if no scripts found
   if not scripts:
       return 'unknown'
   
   return scripts.most_common(1)[0][0]


def transliterate_word(word):
   """
   Transliterate a word using the appropriate library based on its script.
   
   Priority:
   1. If all Latin, return as-is
   2. Route to appropriate library based on primary script
   """
   
   # If all Latin, don't transliterate
   if is_all_latin(word):
       return word
   
   # Detect primary script
   primary_script = detect_primary_script(word)
   
   # Route to appropriate library
   if primary_script == 'hiragana' or primary_script == 'katakana':
       # Japanese kana - use transliter
       return tl.jp(word)
   
   elif primary_script == 'hangul':
       # Korean - use transliter
       return tl.ko(word)
   
   elif primary_script == 'chinese':
       # Chinese - use pypinyin
       return ''.join(lazy_pinyin(word))
   
   elif primary_script == 'arabic':
       # Arabic - use gimeltra
       return gimeltra_tr(word, sc='Arab', to_sc='Latn')
   
   elif primary_script == 'hebrew':
       # Hebrew - use gimeltra
       return gimeltra_tr(word, sc='Hebr', to_sc='Latn')
   
   elif primary_script == 'armenian':
       # Armenian - use transliterate
       return translit(word, 'hy', reversed=True)
   
   elif primary_script == 'georgian':
       # Georgian - use transliterate
       return translit(word, 'ka', reversed=True)
   
   elif primary_script == 'greek':
       # Greek - use transliterate
       return translit(word, 'el', reversed=True)
   
   elif primary_script == 'cyrillic':
       # Cyrillic - use transliter (defaulting to Russian)
       return tl.ru(word)
   
   elif primary_script == 'thai':
       # Thai - use pythainlp
       return romanize(word, engine="royin")
   
   elif primary_script == 'devanagari':
       # Hindi/Sanskrit - use indic-transliteration
       return indic_translit(word, sanscript.DEVANAGARI, sanscript.IAST)
   
   elif primary_script == 'bengali':
       # Bengali - use indic-transliteration
       return indic_translit(word, sanscript.BENGALI, sanscript.IAST)
   
   elif primary_script == 'tamil':
       # Tamil - use indic-transliteration
       return indic_translit(word, sanscript.TAMIL, sanscript.IAST)
   
   elif primary_script == 'telugu':
       # Telugu - use indic-transliteration
       return indic_translit(word, sanscript.TELUGU, sanscript.IAST)
   
   elif primary_script == 'kannada':
       # Kannada - use indic-transliteration
       return indic_translit(word, sanscript.KANNADA, sanscript.IAST)
   
   elif primary_script == 'malayalam':
       # Malayalam - use indic-transliteration
       return indic_translit(word, sanscript.MALAYALAM, sanscript.IAST)
   
   elif primary_script == 'gujarati':
       # Gujarati - use indic-transliteration
       return indic_translit(word, sanscript.GUJARATI, sanscript.IAST)
   
   elif primary_script == 'gurmukhi':
       # Punjabi - use indic-transliteration
       return indic_translit(word, sanscript.GURMUKHI, sanscript.IAST)
   
   elif primary_script == 'oriya':
       # Oriya - use indic-transliteration
       return indic_translit(word, sanscript.ORIYA, sanscript.IAST)
   
   elif primary_script == 'sinhala':
       # Sinhala - use indic-transliteration
       return indic_translit(word, sanscript.SINHALA, sanscript.IAST)
   
   else:
       # Unknown script or 'other' - return original
       return word


def main():
   """Test the transliteration with various scripts"""
   test_words = [
       "Hello",        # Latin
       "café",         # Latin with diacritics
       "こんにちは",      # Japanese hiragana
       "カタカナ",       # Japanese katakana
       "안녕하세요",     # Korean
       "你好",          # Chinese
       "مرحبا",        # Arabic
       "שלום",         # Hebrew
       "Привет",       # Cyrillic
       "Բարև",         # Armenian
       "გამარჯობა",     # Georgian
       "Γεια σας",     # Greek
       "สวัสดี",        # Thai
       "नमस्ते",       # Hindi (Devanagari)
       "নমস্কার",       # Bengali
       "வணக்கம்",      # Tamil
   ]
   
   print("Testing transliteration:")
   print("-" * 50)
   
   for word in test_words:
       primary_script = detect_primary_script(word)
       result = transliterate_word(word)
       print(f"'{word}' ({primary_script}) -> '{result}'")


if __name__ == "__main__":
   main()
