"""
Mail Processing Module
Handles email extraction, parsing, threading, and alias resolution.
"""

from mail_processing.mail import Mail
from mail_processing.mailconverter import MailConverter, EmlxConverter
from mail_processing.alias import alias

__all__ = ['Mail', 'MailConverter', 'EmlxConverter', 'alias']
