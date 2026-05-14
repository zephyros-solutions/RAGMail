"""
Convert Mail objects to LlamaIndex Nodes with structured metadata.

This module provides utilities to convert Mail objects into LlamaIndex TextNode objects,
enabling seamless integration with LlamaIndex vector stores (including Milvus).

Key features:
- Works directly with Mail objects (no JSON dependency)
- Configurable text styles: with/without date, minimal, or content-only
- Structured metadata extraction from Mail fields
- Optional semantic chunking with metadata preservation

Non-breaking: existing text-chunk-based workflows continue unchanged (blob, grep, ES).
LlamaIndex integration only affects Milvus strategy.
"""

from typing import List, Dict, Any, Optional, Literal

from llama_index.core.schema import TextNode, BaseNode

from mail_processing.mail import Mail


class MailNodeConverter:
    """Convert Mail objects to LlamaIndex TextNode objects with metadata."""
        
    @staticmethod
    def _extract_metadata(mail: Mail, additional: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract metadata from Mail object.
        
        Args:
            mail: Mail object
            additional: Additional metadata to merge
            
        Returns:
            Metadata dictionary with sender, subject, date, conversation context
        """
        metadata = {
            "subject": mail.Subject or Mail.NO_SUB,
            "from": mail.From or "Unknown",
            "to": mail.To or Mail.NO_REC,
            "date": mail.Date.isoformat() if mail.Date else "unknown",
            "is_reply": mail.isReply,
            "conversation_id": mail.ConversationID or "unknown",
        }
        
        if additional:
            metadata.update(additional)
        
        return metadata
    
    @staticmethod
    def mail_to_node(
        mail: Mail,
        node_id: Optional[str] = None,
        text_style: Literal['full', 'minimal', 'content_only'] = 'minimal'
    ) -> BaseNode:
        """
        Convert a Mail object to a LlamaIndex TextNode.
        
        Args:
            mail: Mail object to convert
            node_id: Optional custom node ID (defaults to auto-generated)
            text_style: How to format text content:
                - 'full': Include date in preamble 
                  "Il 2006-11-02 13:48:03+01:00 Stefano Bocconi risponde a Federica Cena..."
                - 'minimal': Subject + content, no date 
                  "Subject: Re: It's beautiful!\n\nContent here..."
                - 'content_only': Just email body, no context
                  "Content here..."
        
        Returns:
            TextNode with Mail metadata (sender, subject, date, conversation_id, etc.)
            
        Raises:
            ImportError: If llama-index-core not installed
        """
        
        # Format text based on style preference
        if text_style == 'full':
            # Include full preamble with date (backward compatible with original code)
            date_str = mail.Date.strftime("%Y-%m-%d %H:%M:%S %z") if mail.Date else "unknown date"
            text = Mail.mail_preamble(
                date_str,
                mail.From or "Unknown",
                mail.To or Mail.NO_REC,
                mail.Subject or Mail.NO_SUB,
                mail.Content,
                mail.isReply
            )
        elif text_style == 'minimal':
            # Subject + content only (cleaner for embeddings, date in metadata)
            text = f"Subject: {mail.Subject or Mail.NO_SUB}\n\n{mail.Content}"
        else:  # content_only
            # Pure email body (best for dense semantic embeddings)
            text = mail.Content
        
        metadata = MailNodeConverter._extract_metadata(mail)
        
        node = TextNode(
            text=text,
            metadata=metadata,
            id_=node_id
        )
        
        return node
    
    @staticmethod
    def mails_to_nodes(
        mails: List[Mail],
        start_node_id: int = 0,
        text_style: Literal['full', 'minimal', 'content_only'] = 'minimal'
    ) -> List[BaseNode]:
        """
        Convert multiple Mail objects to LlamaIndex TextNodes.
        
        Args:
            mails: List of Mail objects to convert
            start_node_id: Starting ID for auto-generated node IDs
            text_style: Text formatting style (see mail_to_node for options)
        
        Returns:
            List of TextNode objects with Mail metadata
            
        Raises:
            ImportError: If llama-index-core not installed
        """
        
        nodes = []
        for idx, mail in enumerate(mails):
            node = MailNodeConverter.mail_to_node(
                mail,
                node_id=f"mail_{start_node_id + idx}",
                text_style=text_style
            )
            # Add per-email index for tracking in large batches
            node.metadata['mail_index'] = idx
            nodes.append(node)
        
        return nodes
    
    @staticmethod
    def chunks_to_nodes(
        chunks: List[str],
        mail: Mail,
        start_node_id: int = 0
    ) -> List[BaseNode]:
        """
        Convert email chunks to Nodes with per-chunk + Mail-level metadata.
        
        Useful when emails are already chunked (e.g., by MailConverter.make_chunks()),
        but you want to preserve email context in metadata.
        
        Args:
            chunks: List of text chunks (from MailConverter.make_chunks() or similar)
            mail: Source Mail object (provides metadata)
            start_node_id: Starting ID for auto-generated node IDs
        
        Returns:
            List of TextNode objects (one per chunk) with Mail metadata
            
        Raises:
            ImportError: If llama-index-core not installed
        """
        
        nodes = []
        for chunk_idx, chunk in enumerate(chunks):
            if not chunk or len(chunk.strip()) == 0:
                continue
            
            # Metadata: mail-level info + chunk-specific tracking
            metadata = MailNodeConverter._extract_metadata(
                mail,
                additional={
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                }
            )
            
            node = TextNode(
                text=chunk,
                metadata=metadata,
                id_=f"chunk_{start_node_id + chunk_idx}"
            )
            nodes.append(node)
        
        return nodes
