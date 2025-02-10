from datetime import datetime, timezone
from pathlib import Path
import re
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
          return ' '.join(BeautifulSoup(html_text, "html.parser").stripped_strings)


     @staticmethod
     def handle_replies(content_str):
          
          mailer_pt =  '[\w <>@\.]+'
          ending = '[\n ]+[\s\S]+'
          
          patterns = [ 
                    # rf'\n+(Il \d\d\/\d\d\/\d\d \d\d[:\.]\d\d, )?({mailer_pt}) ha scritto:\n({ending})?',
                    # rf'\n+?(?:Il giorno \d{2}[ \/]\w+[ \/]\d{4}, alle ore \d\d:\d\d, )?({mailer_pt}) ha scritto:\n({ending})?',
                    # rf'\n(?:On \d\d\/\d\d\/\d\d \d\d:\d\d, )?({mailer_pt}) wrote:\n({ending})?',
                    rf'\n?[\w <>\.@\|,\/:]+ha scritto:({ending})?',                    
                    rf'\nScrive [\s\S]+:\n({ending})?',
                    # rf'\n?[\w <>\.@\|,\/:]+wrote:({ending})?',
                    rf'[\n|\.] [\w| |@|\.]+wrote:({ending})?',
                    rf'[\n ]?+-+ ?Original Message ?-+[\n ]({ending})?',
                    rf'\w+ \w+ wrote:({ending})?',
                    rf' (\w+\.)?\w+@\w+\.\w+(\.\w+)? wrote:({ending})?',
                    rf'\w+ \w+ <(\w+\.)?\w+@\w+\.\w+(\.\w+)?> wrote:({ending})?'
          ]
          
          flt_txt = content_str
          
          for pattern in patterns:
               p = re.compile(pattern)
               match = p.search(flt_txt)
          
               if match != None:
                    flt_txt = re.sub(pattern, '', flt_txt, flags=re.MULTILINE)
                    # print(f"Matched pattern {match}, {match.group(1)}, {match.group(2)}")
                    # breakpoint()
                    # quoted_mail = RAGMail('quoted_' + mail.msg_filename)
                    # quoted_mail.setDate(mail.Date)
                    # quoted_mail.setFrom(m.group(1))
                    # quoted_mail.setTo(mail.From)
                    # quoted_mail.setContent(m.group(2), True)
                    # pattern = r'Re: '
                    # quoted_mail.setSubject(re.sub(pattern, '', mail.Subject).strip())
                    # flt_txt = re.sub(pattern, '', flt_txt, flags=re.MULTILINE)
                    # return quoted_mail
                    
          
          # if flt_txt.find('Original Message') != -1:
          #      breakpoint()
          
          return flt_txt

     
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
          for key,value in patterns.items():
               flt_txt = re.sub(key, value, flt_txt, flags=re.IGNORECASE)

          # Normalize whitespace to single spaces, strip leading/trailing whitespace
          flt_txt = re.sub(r'\s+', ' ', flt_txt).strip()

          return flt_txt


     @classmethod
     def norm_mailer(cls, mailer):
          
          # In case there are more mailers separated by commas
          mailer = mailer.split(',')[0]
          
          # Is the mailer name + email address?
          pattern = r'([\w\s@\.\\\|\-\']+)(?: <\S+@\S+>)?'
          p = re.compile(pattern)
          match = p.search(mailer)
          if match == None:
               flt_mailer= mailer
          else:
               flt_mailer = match.group(1)
          
          # Is the mailer name surrendered by quotations or spaces ["'] ?
          pattern = r'^[\'" ]+|[\'" ]+$'
          flt_mailer = re.sub(pattern, '', flt_mailer)
          # breakpoint()
          
          # Is the lenght of the mailer suspiciously long?
          if len(flt_mailer) > cls.MAX_MAILER_LEN:
               raise Exception(f"Mailer too long: {flt_mailer}")

          # print(f"Mailer: {mailer}, filt: __{flt_mailer}__")
          # breakpoint()
          # Resolve alias
          if not flt_mailer in alias.keys():
               found = False
               for key in alias.keys():
                    if flt_mailer in alias[key]:
                         flt_mailer = key
                         found = True
               if not found:
                    print(f"No match for {flt_mailer}")
                    # breakpoint()
          
          # print(flt_mailer)
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
          reply.isReplay = True
          cnt = reply.get_content()
          self.Content = self.Content + '\n' + cnt