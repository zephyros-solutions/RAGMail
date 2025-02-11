import os
import re
from datetime import datetime, timezone
import emlx
import glob
from pathlib import Path

from email import policy
from email.parser import BytesParser
from tqdm import tqdm

from mail import Mail
from globals import USERNAME

class MailConverter:

     # CHUNKS_FILE = "chunks.txt"
     OUT_DIR = './proc_mails'
     OUT_FMT = r'%d-%m-%Y'
     
     def __init__(self,mailbox:str, doThreads:bool, start_date:datetime, end_date:datetime):
          self.folder = {}
          self.proc_folder = {}
          self.start_date = start_date
          self.end_date = end_date
          if mailbox == None:
               raise Exception(f"Mailbox needs to be specified")
          self.mailbox = mailbox
          self.doThreads = doThreads
          self.mailsId = f"{self.mailbox}_{'T' if doThreads else 'NT'}_{datetime.strftime(start_date,self.OUT_FMT)}_{datetime.strftime(end_date,self.OUT_FMT)}"

     
     def add_convrs(self, mail):
          key = f"{mail.CoversationID}"
          if key in self.folder:
               self.folder[key].append(mail)
          else:
               self.folder[key] = [mail]

     def proc_msgs(self, doThreads):
          for key in self.folder.keys():
               values = self.folder[key]
               if len(values) == 1:
                    if key in self.proc_folder:
                         raise Exception(f'Key {key} already exists in processed emails')
                    self.proc_folder[key] = values[0]
               else:
                    values.sort(key=lambda x: x.Date, reverse=False)
                    # breakpoint()
                    mail = values.pop(0)
                    for i,reply in enumerate(values):
                         if doThreads:
                              mail.addReply(reply)
                         else:
                              reply.isReply=True
                              new_key = f'{key}_{i}'
                              if new_key in self.proc_folder:
                                   raise Exception(f'Key {new_key} already exists in processed emails')
                              self.proc_folder[new_key] = reply
                    if key in self.proc_folder:
                         raise Exception(f'Key {key} already exists in processed emails')
                    self.proc_folder[key] = mail
                    # breakpoint()

     def save_msgs(self):
          self.mail_out_dir = f"{self.OUT_DIR}_{self.mailsId}"
          if not (p:=Path(self.mail_out_dir)).is_dir():
               p.mkdir(parents=True, exist_ok=True) 

          for key in self.proc_folder.keys():
               mail = self.proc_folder[key]
               mail.save(self.mail_out_dir)
                    # breakpoint()


# class EmlConverter(MailConverter):
#      '''
#           At the moment this is not at the level of the Emlx converter.
#      '''

#      def __init__(self, mail_in_dir):
#           super().__init__()
#           self.mail_in_dir = mail_in_dir

#      def read_mails(self, mail_out_dir):

#           if os.path.samefile(self.mail_in_dir, mail_out_dir):
#                raise Exception("In and Out directories cannot be the same")
          
#           nr_mails = 0
#           nr_text_mails = 0
#           nr_html_mails = 0
#           nr_html_text_mails = 0
          
#           filepaths = super.mail_paths(self.mail_in_dir, 'eml')

#           for filepath in tqdm(filepaths,desc="Processing eml mails"):
#                # print(f"file: {filepath}")
#                # mail = RAGMail(os.path.basename(filepath))
#                mail = Mail(os.path.basename(filepath))
#                nr_mails = nr_mails + 1
#                with open(filepath, 'rb') as fp:  # select a specific email file from the list
#                     # name = fp.name # Get file name
#                     m = BytesParser(policy=policy.default).parse(fp)


#                if (field := m.get("Date")) != None:
#                     mail.setDate(field)
#                else:
#                     raise Exception(f"Message {filepath} has no Date")
               
#                mail.setFrom(m.get("From"))
          
#                mail.setTo(m.get("To"))
               
#                mail.setSubject(m.get("Subject"))

#                # body = m.get_body(preferencelist=('plain')).get_content()
#                cnt_types = [part.get_content_type() for part in m.walk()]
               
#                if 'text/plain' in cnt_types and 'text/html' in cnt_types:
#                     nr_html_text_mails = nr_html_text_mails + 1
#                     body = m.get_body('plain').get_content()
#                     mail.setContent(body, False)
#                elif 'text/plain' in cnt_types:
#                     nr_text_mails = nr_text_mails + 1
#                     body = m.get_body('plain').get_content()
#                     mail.setContent(body, False)
#                elif 'text/html' in cnt_types:
#                     nr_html_mails = nr_html_mails + 1
#                     body = m.get_body('html').get_content()
#                     mail.setContent(body, True)
#                else:
#                     raise Exception(f"No content for message {filepath}")

#                # print(cnt_types)
#                # print(body)
#                # breakpoint()

#                mail.save(mail_out_dir)
             


#           print(f"Mails: {nr_mails}, of which {nr_text_mails} text, {nr_html_text_mails} both text and html, and {nr_html_mails} html")


class EmlxConverter(MailConverter):
     '''
          m.headers
          {
             'X-Mozilla-Keys': '',
             'Message-ID': '<45EC23F2.4010701@di.unito.it>',
             'Date': 'Mon, 05 Mar 2007 15:06:42 +0100',
             'From': 'From',
             'User-Agent': 'Thunderbird 1.5.0.10 (Windows/20070221)',
             'MIME-Version': '1.0',
             'To': 'Recipient',
             'Subject': 'Re: ',
             'References': '<1173096974.45ec0a0e22b3f@www.di.unito.it>',
             'In-Reply-To': '<1173096974.45ec0a0e22b3f@www.di.unito.it>',
             'Content-Type': 'text/plain; charset=ISO-8859-1; format=flowed',
             'Content-Transfer-Encoding': '8bit'
          }
          m.plist
          {
             'conversation-id': 16220,
             'date-last-viewed': 0,
             'date-received': 1173100002,
             'flags':
                 {
                  'read': True,
                  'priority_level': 3
                 },
             'remote-id': '695'
          }
          m.flags:
          {
             'read': True,
             'priority_level': 3
          }
     '''


     def __init__(self, mailbox:str, doThreads:bool, start_date:datetime, end_date:datetime):
          super().__init__(mailbox=mailbox, doThreads=doThreads, start_date=start_date, end_date=end_date)
          


     def read_mails(self):
          nr_mails = 0
          nr_text_mails = 0
          nr_html_mails = 0
          nr_html_text_mails = 0
          mismatch = 0

          filepaths = [file for file in glob.iglob(f"/Users/{USERNAME}/Library/Mail/**/{self.mailbox}/**/*.emlx", recursive=True)]
          for filepath in tqdm(filepaths,desc="Processing emlx emails"):
               # print(f"file: {filepath}")
               nr_mails = nr_mails + 1
               m = emlx.read(filepath, encoding='utf-8')

               mail = Mail(os.path.basename(filepath))
               # mail = RAGMail()
               # breakpoint()
               
               mail_date = Mail.parse_date(m.headers["Date"])
               if mail_date < self.start_date or mail_date > self.end_date:
                    # breakpoint()
                    continue

               mail.setDate(mail_date)
               # Try to fix strange mismatch in dates
               if not datetime.fromtimestamp(m.plist['date-received']).replace(tzinfo=timezone.utc) == mail.Date:
                    if not datetime.fromtimestamp(m.plist['date-received']).replace(tzinfo=mail.Date.tzinfo) == mail.Date:
                         # breakpoint()
                         delta = min(datetime.fromtimestamp(m.plist['date-received']).replace(tzinfo=timezone.utc) - mail.Date, datetime.fromtimestamp(m.plist['date-received']).replace(tzinfo=mail.Date.tzinfo) - mail.Date)
                         mail.Date = mail.Date + delta
                         mismatch = mismatch + 1
                    # breakpoint()
               
               mail.setFrom(m.headers["From"])
          
               if "To" in m.headers:
                    mail.setTo(m.headers["To"])
               else:
                    mail.setTo(None)
               
               if "Subject" in m.headers:
                    mail.setSubject(m.headers["Subject"])
               else:
                    mail.setSubject(None)

               if m.text != None and m.html != None:
                    # breakpoint()
                    nr_html_text_mails = nr_html_text_mails + 1

               if m.text != None:
                    nr_text_mails = nr_text_mails + 1
                    mail.setContent(m.text, False)
               elif m.html != None:
                    nr_html_mails = nr_html_mails + 1
                    mail.setContent(m.html, True)
               else:
                    raise Exception(f'Mail {m.headers["Subject"]} has no content')


               mail.setConversationID(m.plist['conversation-id'])
               # quoted_mail = BaseMailConv.handle_replies(mail)
               # breakpoint()
               # mail.save(mail_out_dir)
               self.add_convrs(mail)
               # quoted_mail.save(mail_out_dir)
               

          self.proc_msgs(self.doThreads)
          # self.save_msgs()
          print(f"Mails: {nr_mails}, of which {nr_text_mails} text, {nr_html_text_mails} both text and html, and {nr_html_mails} html")
          print(f"Time mismatches: {mismatch}")

     def mail_paths(self, ext=Mail.EXT):
          mailpaths = [file for file in glob.iglob(f"{self.mail_out_dir}/*.{ext}", recursive=False)]
          return mailpaths

     def make_blob(self):
          # breakpoint()
          blob = ""
          for key,value in self.proc_folder.items():
               # breakpoint()
               blob = blob + "\n" + value.get_content()
          return blob

     def msgs_array(self):
          msgs = []

          for key,mail_cnt in self.proc_folder.items():
              msgs.append(mail_cnt) 
              
          return msgs

     def make_chunks(self, max_chunk_len, max_chunk_excess):
          text_chunks = []

          for key,mail_cnt in self.proc_folder.items():
               current_chunk = ""

               # Split text into sentences while preserving punctuation
               # sentences = re.split(r'(?<=[.!?]) +', file_cnt)
               sentences = re.split(r'(?<=[.!?;])(?: )*', mail_cnt)
               
               if len(sentences) == 1 and len(sentences[0]) > max_chunk_excess * max_chunk_len:
                    sentences = re.split(r'(?<=[.!?; ])(?: )*', mail_cnt)
                    # breakpoint()

               for sentence in sentences:                    
                    if len(current_chunk) + len(sentence) + 1 < max_chunk_len:
                         current_chunk += sentence + " "
                    else:
                         text_chunks.append(current_chunk)
                         current_chunk = sentence + " "
               if len(current_chunk) > 0:
                    text_chunks.append(current_chunk.strip())

               # chunks_path = Path(chunks_dir, cls.CHUNKS_FILE)
               # # breakpoint()
               # with open(chunks_path, "a", encoding="utf-8") as chunks_file:
               #      for chunk in text_chunks:
               #           chunks_file.write(chunk.strip() + "\n"
          return text_chunks
