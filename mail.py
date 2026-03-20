from datetime import datetime, timezone
from pathlib import Path
import re
import html
import quopri
from email.utils import getaddresses

from bs4 import BeautifulSoup

from alias import alias
from vocab import mail_preamble


class Mail:
     EXT = 'txt'
     NO_SUB = "(No Subject)"
     NO_REC = "(No Recipient)"
     IN_TM_FMT = "%a, %d %b %Y %H:%M:%S %z"
     OUT_TM_FMT = "%d_%m_%Y_%H:%M:%S_%z"
     MAX_MAILER_LEN = 40


     @staticmethod
     def process_html(html_text):
          if BeautifulSoup is not None:
               return ' '.join(BeautifulSoup(html_text, "html.parser").stripped_strings)

          # Fallback when BeautifulSoup is not installed: strip tags crudely.
          text = re.sub(r'<[^>]+>', ' ', html_text)
          return html.unescape(text)


     @staticmethod
     def strip_signatures(text):
          """Remove common email signatures and footers.
          
          Detects and removes:
          - Standard signature separator (-- on its own line)
          - Lines with contact info (email, phone, fax, URLs)
          - Common signature patterns (titles, company info)
          - Social media links and legal disclaimers
          """
          if not text:
               return text
          
          lines = text.split('\n')
          output_lines = []
          in_signature = False
          
          for i, line in enumerate(lines):
               # Standard signature separator
               if re.match(r'^\s*--\s*$', line):
                    in_signature = True
                    break
               
               # Skip if already in signature block
               if in_signature:
                    continue
               
               output_lines.append(line)
          
          text = '\n'.join(output_lines).rstrip()
          
          # Also strip common footer patterns that don't use separators
          signature_patterns = [
               r'(?i)^\s*[a-z\s\.]+\s+\|\s+[a-z\s\.]+\s*$',  # "Name | Title" pattern
               r'(?i)^best regards,?\s*$',
               r'(?i)^kind regards,?\s*$',
               r'(?i)^cordiali saluti,?\s*$',  # Italian: "Kind regards"
               r'(?i)^distinti saluti,?\s*$',  # Italian: "Best regards"
               r'(?i)^grazie,?\s*$',  # Italian: "Thank you"
               r'(?i)^buongiorno,?\s*$',  # Italian: "Good morning" (when used as closing)
          ]
          
          # Trim trailing lines that match signature patterns
          text_lines = text.split('\n')
          
          # Common signature phrases that indicate the start of a signature block
          closing_phrases = [
               'best regards', 'kind regards', 'sincerely', 'regards',
               'cordiali saluti', 'distinti saluti', 'grazie',
               'thanks', 'thank you', 'buongiorno', 'all the best'
          ]
          
          # Remove lines after a known closing phrase
          removed_lines = 0
          for i in range(len(text_lines) - 1, -1, -1):
               line_lower = text_lines[i].lower().strip()
               
               # Check if this line is a closing phrase
               if any(phrase in line_lower for phrase in closing_phrases):
                    # Remove this line and all following lines
                    text_lines = text_lines[:i]
                    removed_lines = len(text_lines) - i
                    break
               
               # Also check regex patterns for non-textual patterns
               for pattern in signature_patterns:
                    pattern_clean = pattern.replace('(?i)', '')
                    if re.match(pattern_clean, line_lower, re.IGNORECASE):
                         text_lines = text_lines[:i]
                         removed_lines = len(text_lines) - i
                         break
               
               if removed_lines > 0:
                    break
          
          #Also check for -- separator
          for i in range(len(text_lines) - 1, -1, -1):
               if re.match(r'^\s*--\s*$', text_lines[i]):
                    text_lines = text_lines[:i]
                    break
          
          return '\n'.join(text_lines).strip()

     @staticmethod
     def handle_replies(content_str):
          """Strip quoted/forwarded blocks and common reply markers.

          The goal is to keep only the new text written in the message while
          removing the parts that are quoted from previous messages.
          """

          if content_str is None:
               return ''

          # Normalize line endings for reliable regex matching
          text = content_str.replace('\r\n', '\n').replace('\r', '\n')

          # Trim at common reply/forward boundaries
          boundaries = [
               r'(?m)^[\s>-]*[-]{2,}\s*Original Message\s*[-]{2,}.*$',
               r'(?m)^[\s>-]*[-]{2,}\s*Forwarded message\s*[-]{2,}.*$',
               r'(?m)^On .+ wrote:$',
               r'(?m)^[^\n]+ wrote:$',
               r'(?m)^[^\n]+<[^>]+> wrote:$',
               # Sometimes the "X wrote:" marker is in-line (no newline)
               r'(?i)\b[\w\.\-]+(?: [\w\.\-]+){0,4}\s+wrote:',
               r'(?i)\b[\w\.\-]+(?: [\w\.\-]+){0,4}\s+ha scritto:',
               r'(?m)^[\s>-]*[\w <>\.@\|,\/]+ha scritto:$',
               r'(?m)^[\s>-]*Scrive .+:$',
          ]

          cutoff = len(text)
          for b in boundaries:
               m = re.search(b, text)
               if m and m.start() < cutoff:
                    cutoff = m.start()
          text = text[:cutoff]

          # Remove quoted lines (e.g. lines prefixed with '>' or '|')
          kept_lines = []
          for line in text.splitlines():
               if re.match(r'^\s*[>\|]+', line):
                    continue
               kept_lines.append(line)
          text = '\n'.join(kept_lines)

          # Remove typical signature separators
          text = re.split(r'(?m)^\s*(--|__|==)\s*$', text)[0]

          # Collapse redundant blank lines
          text = re.sub(r'\n{3,}', '\n\n', text).strip()

          return text

     
     @classmethod
     def parse_date(cls, date_str, fmt=IN_TM_FMT):
          try:
               date_parsed = datetime.strptime(date_str,fmt)
          except ValueError as v:
               # breakpoint()
               if len(v.args) > 0 and v.args[0].startswith('unconverted data remains: '):
                    pattern = r"( \(.*\))"
                    date_str = re.sub(pattern, '', date_str)
                    
                    return cls.parse_date(date_str)
               if len(v.args) > 0 and 'does not match format' in v.args[0]:
                    fmt = "%d %b %Y %H:%M:%S %z"
                    date_parsed = datetime.strptime(date_str,fmt)
                    # breakpoint()
               else:
                    raise Exception(f"Unable to parse {date_str}")
                    # print(f"Unable to parse {date_str}")
                    # date_parsed = datetime.today()
          
          return date_parsed
     
     @staticmethod
     def filter_text(text):
          '''
               cat codes.txt | cut  -f 2,3 | sed "s/\(.\)\t\([a-z0-9][a-z0-9]\) \([a-z0-9][a-z0-9]\)/r'=\2=\3' : '\1',/g"
          '''
          patterns = { r'\xa0' : ' ',
                      r'=20' : ' ',
                      r'=3D' : '',
                      r'=EC' : 'í',
                      r'=B9' : '\'',
                      r'=E8' : 'é',
                      r'=E2=80=99' : '\'',
                      r'=c2=a1' : '¡', r'=c2=a2' : '¢', r'=c2=a3' : '£', r'=c2=a4' : '¤', r'=c2=a5' : '¥', r'=c2=a6' : '¦',
                      r'=c2=a7' : '§', r'=c2=a8' : '¨', r'=c2=a9' : '©', r'=c2=aa' : 'ª', r'=c2=ab' : '«', r'=c2=ac' : '¬',
                      r'=c2=ad' : '­', r'=c2=ae' : '®', r'=c2=af' : '¯', r'=c2=b0' : '°', r'=c2=b1' : '±', r'=c2=b2' : '²',
                      r'=c2=b3' : '³', r'=c2=b4' : '´', r'=c2=b5' : 'µ', r'=c2=b6' : '¶', r'=c2=b7' : '·', r'=c2=b8' : '¸',
                      r'=c2=b9' : '¹', r'=c2=ba' : 'º', r'=c2=bb' : '»', r'=c2=bc' : '¼', r'=c2=bd' : '½', r'=c2=be' : '¾',
                      r'=c2=bf' : '¿', r'=c3=80' : 'À', r'=c3=81' : 'Á', r'=c3=82' : 'Â', r'=c3=83' : 'Ã', r'=c3=84' : 'Ä',
                      r'=c3=85' : 'Å', r'=c3=86' : 'Æ', r'=c3=87' : 'Ç', r'=c3=88' : 'È', r'=c3=89' : 'É', r'=c3=8a' : 'Ê',
                      r'=c3=8b' : 'Ë', r'=c3=8c' : 'Ì', r'=c3=8d' : 'Í', r'=c3=8e' : 'Î', r'=c3=8f' : 'Ï', r'=c3=90' : 'Ð',
                      r'=c3=91' : 'Ñ', r'=c3=92' : 'Ò', r'=c3=93' : 'Ó', r'=c3=94' : 'Ô', r'=c3=95' : 'Õ', r'=c3=96' : 'Ö',
                      r'=c3=97' : '×', r'=c3=98' : 'Ø', r'=c3=99' : 'Ù', r'=c3=9a' : 'Ú', r'=c3=9b' : 'Û', r'=c3=9c' : 'Ü',
                      r'=c3=9d' : 'Ý', r'=c3=9e' : 'Þ', r'=c3=9f' : 'ß', r'=c3=a0' : 'à', r'=c3=a1' : 'á', r'=c3=a2' : 'â',
                      r'=c3=a3' : 'ã', r'=c3=a4' : 'ä', r'=c3=a5' : 'å', r'=c3=a6' : 'æ', r'=c3=a7' : 'ç', r'=c3=a8' : 'è',
                      r'=c3=a9' : 'é', r'=c3=aa' : 'ê', r'=c3=ab' : 'ë', r'=c3=ac' : 'ì', r'=c3=ad' : 'í', r'=c3=ae' : 'î',
                      r'=c3=af' : 'ï', r'=c3=b0' : 'ð', r'=c3=b1' : 'ñ', r'=c3=b2' : 'ò', r'=c3=b3' : 'ó', r'=c3=b4' : 'ô',
                      r'=c3=b5' : 'õ', r'=c3=b6' : 'ö', r'=c3=b7' : '÷', r'=c3=b8' : 'ø', r'=c3=b9' : 'ù', r'=c3=ba' : 'ú',
                      r'=c3=bb' : 'û', r'=c3=bc' : 'ü', r'=c3=bd' : 'ý', r'=c3=be' : 'þ', r'=c3=bf' : 'ÿ'
          }
          

          flt_txt = text

          # Decode quoted-printable sequences (e.g. =20, =E8)
          try:
               flt_txt = quopri.decodestring(flt_txt.encode('utf-8')).decode('utf-8', errors='replace')
          except Exception:
               pass

          # Decode HTML entities (common in HTML parts)
          flt_txt = html.unescape(flt_txt)

          for key,value in patterns.items():
               flt_txt = re.sub(key, value, flt_txt, flags=re.IGNORECASE)

          # Normalize whitespace but preserve paragraph structure
          flt_txt = re.sub(r'[ \t]+', ' ', flt_txt)
          flt_txt = re.sub(r'\n{3,}', '\n\n', flt_txt)
          flt_txt = flt_txt.strip()

          # Remove email signatures and footers
          flt_txt = Mail.strip_signatures(flt_txt)

          return flt_txt


     @classmethod
     def norm_mailer(cls, mailer):
          """Normalize mailer strings and resolve aliases.

          This attempts to parse an address/recipient string (e.g. "Foo Bar <foo@bar.com>")
          into a canonical name using the alias map.
          """

          if not mailer:
               return Mail.NO_REC

          # Minimal normalization: strip surrounding angle brackets and whitespace
          # But keep [mailto:...] format for initial matching, since variants may use it
          mailer_minimal = mailer.strip().strip('<>').strip()
          
          # Strip email metadata at the end (e.g., " On", " wrote:") 
          mailer_minimal = re.sub(r'\s+(On|wrote|wrote:|Scrive)\s*$', '', mailer_minimal, flags=re.IGNORECASE)
          
          # First, try to match the minimally-normalized string directly against aliases
          # This handles plain names and email formats before heavier normalization
          mailer_lower = mailer_minimal.lower().casefold()
          for canonical, variants in alias.items():
               if canonical.casefold() == mailer_lower:
                    return canonical
               for v in variants:
                    if v.casefold() == mailer_lower:
                         return canonical

          # Now apply fuller normalization for second-pass matching
          # Handle non-standard [mailto:...] format by converting to standard <...> format
          mailer_normalized = re.sub(r'\[mailto:([^\]]+)\]', r'<\1>', mailer_minimal)
          mailer_normalized = re.sub(r'=20$', '', mailer_normalized)

          # Support lists of recipients; pick the first one.
          addr = getaddresses([mailer_normalized])
          name, email = addr[0] if addr else ('', '')

          flt_mailer = (name or email or '').strip()
          flt_mailer = re.sub(r'^[\'" ]+|[\'" ]+$', '', flt_mailer)
          flt_mailer = re.sub(r'\s+', ' ', flt_mailer).strip()

          if not flt_mailer and email:
               flt_mailer = email

          # If the string is very long, fall back to the email address
          if len(flt_mailer) > cls.MAX_MAILER_LEN and email:
               flt_mailer = email

          if not flt_mailer:
               flt_mailer = mailer_normalized

          flt_lower = flt_mailer.casefold()
          email_lower = email.casefold() if email else None

          # Try exact matches on the parsed result
          for canonical, variants in alias.items():
               if canonical.casefold() == flt_lower or (email_lower and canonical.casefold() == email_lower):
                    return canonical
               for v in variants:
                    if v.casefold() == flt_lower or (email_lower and v.casefold() == email_lower):
                         return canonical

          # Nothing matched; return normalized mailer
          return flt_mailer

     def __init__(self, orig_file):
          
          p = Path(orig_file)
          self.orig_file = f"{p.stem}.{Mail.EXT}"
          self.To = None
          self.From = None
          self.Date = None
          self.Subject = None
          self.Content = None
          self.CoversationID = None
          self.isReply = False


     def __str__(self):
          return self.get_content()
     
     def get_content(self):
          cnt = mail_preamble( self.Date, self.From, self.To, self.Subject, self.Content, self.isReply)
          
          return cnt

     def save(self, mail_out_dir):
          # if self.From != None:
               # out_date = self.Date.strftime(BaseMailConv.OUT_TM_FMT)
               # filename = f'{mail_out_dir}/' + slugify(f'{self.From}_{self.To}_{self.Subject}_{out_date}')
          filename = Path(mail_out_dir,self.orig_file)
          
          file_cnt = self.get_content()
          
          with open(filename, 'w') as f:
               f.write(file_cnt)
          return
     
     def setDate(self, date_txt):
          if type(date_txt) == datetime:
               self.Date = date_txt
          else:
               self.Date = Mail.parse_date(date_txt)

     def setFrom(self, from_text):
          self.From = Mail.norm_mailer(from_text)

     def setTo(self, to_text):
          if to_text != None:
               self.To = Mail.norm_mailer(to_text)
          else:
               self.To = Mail.NO_REC

     def setSubject(self, subject_text):
          if subject_text != None and subject_text != "":
               self.Subject = subject_text
          else:
               self.Subject = Mail.NO_SUB

     def setContent(self, orig_content, isHTML):
          if orig_content != None:
               # if orig_content.find('fermento') != -1:
               #      breakpoint()
               if isHTML:
                    try:
                         content_str = Mail.process_html(orig_content)
                    except Exception as e:
                         raise Exception(f"Html {orig_content} not parsable, error {e}")
               else:
                    content_str = orig_content
          else:
               raise Exception(f"Content of {self.Subject} cannot be empty")
                              
          content_str = Mail.filter_text(content_str)

          self.Content = Mail.handle_replies(content_str)
          # breakpoint()
          

     def setConversationID(self, id):
          self.CoversationID = id

     def addReply(self, reply):
          reply.isReply = True
          cnt = reply.get_content()
          self.Content = self.Content + '\n' + cnt