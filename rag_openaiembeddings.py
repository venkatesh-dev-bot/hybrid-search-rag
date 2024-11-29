import json
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from typing import List, Dict, Tuple
import numpy as np
import os
from dotenv import load_dotenv
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
load_dotenv()

class ImprovedRAGSystem:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_model = OpenAIEmbeddings(
            openai_api_key=self.openai_api_key,
            model="text-embedding-3-large"  # Using the latest embedding model
        )
        self.chat_model = ChatOpenAI(
            model="gpt-4o-mini",  # Using the latest GPT-4 model
            temperature=0,
            openai_api_key=self.openai_api_key
        )
        self.vectorstore = None
        self.documents = []
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3)  # Include phrases up to 3 words
        )
        
        # Enhanced mathematical prompt template
        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a mathematical expert specializing in vectors and 3D geometry. 
            Analyze the following context and question carefully to provide a detailed, accurate answer.

            Context:
            {context}

            Question: {question}

            Instructions:
            1. Carefully analyze the question's requirements
            2. Use only information present in the context
            3. For theoretical concepts:
               - Provide clear definitions
               - Explain underlying principles
               - Give examples when possible
            4. For calculations:
               - Show step-by-step solutions
               - Explain each step
               - Use proper mathematical notation
            5. If multiple approaches exist, mention them
            6. If any part of the question cannot be answered from the context, explicitly state so
            7. Don't provide latex code in your answer

            Answer:"""
        )

    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing."""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep mathematical symbols
        text = re.sub(r'[^a-z0-9\s+\-*/=()<>{}[\]|\\.,\'^]', ' ', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Improved text chunking with overlap."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def prepare_documents_from_json(self, json_file_path: str) -> List[Document]:
        """Enhanced document preparation with better chunking and metadata."""
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        documents = []
        
        # Process exercises with improved chunking
        if "exercises" in data:
            for page_num, page_data in data["exercises"].items():
                question = "\n".join(page_data.get("question", []))
                solution = "\n".join(page_data.get("solution", []))
                
                # Preprocess text
                processed_text = self.preprocess_text(question + "\n" + solution)
                
                # Create chunks with overlap
                chunks = self.chunk_text(processed_text)
                
                for i, chunk in enumerate(chunks):
                    documents.append(Document(
                        page_content=chunk,
                        metadata={
                            "title": f"Exercise {page_num} - Part {i+1}",
                            "type": "exercise",
                            "page": page_num,
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        }
                    ))

        # Process concepts with improved chunking
        if "concepts" in data:
            for topic, topic_data in data["concepts"].items():
                content = "\n".join(topic_data.get("content", []))
                processed_text = self.preprocess_text(content)
                chunks = self.chunk_text(processed_text)
                
                for i, chunk in enumerate(chunks):
                    documents.append(Document(
                        page_content=chunk,
                        metadata={
                            "title": f"{topic} - Part {i+1}",
                            "type": "concept",
                            "diagrams": topic_data.get("diagrams", []),
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        }
                    ))

        self.documents = documents
        return documents

    def create_vectorstore(self) -> None:
        """Create enhanced FAISS vectorstore with better indexing."""
        texts = [doc.page_content for doc in self.documents]
        metadata = [doc.metadata for doc in self.documents]
        
        # Create FAISS index with improved parameters
        self.vectorstore = FAISS.from_texts(
            texts,
            self.embedding_model,
            metadatas=metadata,
            normalize_L2=True  # Normalize vectors for better similarity search
        )

    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Improved semantic search with better scoring."""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized")
        
        # Get semantic search results
        results = self.vectorstore.similarity_search_with_score(query, k=top_k*2)  # Get more results initially
        
        # Improved score normalization and filtering
        max_score = max(score for _, score in results) if results else 1.0
        normalized_results = [
            (doc, 1 - (score/max_score))  # Convert distance to similarity score
            for doc, score in results
        ]
        
        # Sort by score and take top_k
        normalized_results.sort(key=lambda x: x[1], reverse=True)
        return normalized_results[:top_k]

    def keyword_search(self, query: str, documents: List[Document]) -> List[Document]:
        """Enhanced keyword search with TF-IDF scoring."""
        # Prepare corpus for TF-IDF
        corpus = [doc.page_content for doc in documents]
        self.tfidf_vectorizer.fit(corpus)
        
        # Transform query and documents
        query_vector = self.tfidf_vectorizer.transform([query])
        doc_vectors = self.tfidf_vectorizer.transform(corpus)
        
        # Calculate similarities
        similarities = (query_vector * doc_vectors.T).toarray()[0]
        
        # Sort documents by similarity
        doc_scores = list(zip(documents, similarities))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in doc_scores[:5]]

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Document]:
        """Improved hybrid search with weighted combination."""
        # Get results from both search methods
        semantic_results = self.semantic_search(query, top_k=top_k)
        keyword_results = self.keyword_search(query, self.documents)
        
        # Create a scoring system that combines both results
        combined_scores = {}
        
        # Add semantic search scores
        for doc, score in semantic_results:
            combined_scores[doc.page_content] = {
                'doc': doc,
                'score': score * 0.7  # Weight for semantic search
            }
        
        # Add keyword search scores
        for i, doc in enumerate(keyword_results):
            score = 1 - (i / len(keyword_results))  # Normalize score based on position
            if doc.page_content in combined_scores:
                combined_scores[doc.page_content]['score'] += score * 0.3  # Weight for keyword search
            else:
                combined_scores[doc.page_content] = {
                    'doc': doc,
                    'score': score * 0.3
                }
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return [item['doc'] for item in sorted_results[:top_k]]

    def create_qa_chain(self, retrieved_docs: List[Document]) -> RetrievalQA:
        """Create a QA chain with the retrieved documents."""
        # Create a new retriever from the retrieved documents
        retriever = FAISS.from_documents(
            retrieved_docs, 
            self.embedding_model
        ).as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # Create the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.chat_model,
            chain_type="stuff",  # This combines all documents into one context
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": self.qa_prompt,
                "verbose": True
            }
        )

        return qa_chain

    def query(self, query: str) -> Dict:
        """Process a query and return the answer with sources."""
        try:
            # Get relevant documents using hybrid search
            retrieved_docs = self.hybrid_search(query)
            
            if not retrieved_docs:
                return {
                    "answer": "I don't have enough information to answer that question.",
                    "sources": []
                }

            # Create enhanced context with metadata
            context_parts = []
            for doc in retrieved_docs:
                # Add document metadata
                context_parts.append(f"\nSource: {doc.metadata.get('title', 'Unknown')}")
                # Add content
                context_parts.append(doc.page_content)
                # Add diagrams if available
                if doc.metadata.get('diagrams'):
                    context_parts.append("\nRelevant diagrams:")
                    context_parts.extend(doc.metadata['diagrams'])
                context_parts.append("-" * 40)

            context = "\n".join(context_parts)

            # Create QA chain with retrieved documents
            qa_chain = self.create_qa_chain(retrieved_docs)
            
            # Get response
            result = qa_chain({"query": query, "context": context})
            
            # Extract answer and sources
            answer = result.get("result", "No answer found.")
            # sources = [doc.metadata.get('title', 'Unknown') for doc in retrieved_docs]
            sources = [retrieved_docs[0].metadata["title"]]  # Only return first source
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return {
                "answer": f"An error occurred while processing your query: {str(e)}",
                "sources": []
            }

# Usage
if __name__ == "__main__":

    # 2. Define the chunks
    # chunks = {
    #     "Distance between Two Points": {
    #         "content": [
    #             "Let \\( P(x_1, y_1, z_1) \\) and \\( Q(x_2, y_2, z_2) \\) be two points referred to a system of rectangular axes \\( OX \\), \\( OY \\) and \\( OZ \\). Through the points \\( P \\) and \\( Q \\) draw planes parallel to the coordinate planes so as to form a rectangular parallelepiped with one diagonal \\( PQ \\).",
    #             "Now, since \\( \\angle PAQ \\) is a right angle, it follows that in triangle \\( PAQ \\),",
    #             "\\[ PQ^2 = PA^2 + AQ^2 \\tag{i} \\]",
    #             "Also, triangle \\( ANQ \\) is right-angled with \\( \\angle ANQ \\) being the right angle. Therefore,",
    #             "\\[ AQ^2 = AN^2 + NQ^2 \\tag{ii} \\]",
    #             "From (i) and (ii), we have",
    #             "\\[ PQ^2 = PA^2 + AN^2 + NQ^2 \\]",
    #             "Now",
    #             "\\[ PA = y_2 - y_1, AN = x_2 - x_1 \\text{ and } NQ = z_2 - z_1 \\]",
    #             "Hence,",
    #             "\\[ PQ^2 = (x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2 \\]",
    #             "\\[ PQ = \\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2} \\]",
    #             "This gives us the distance between two points \\( (x_1, y_1, z_1) \\) and \\( (x_2, y_2, z_2) \\).",
    #             "In particular, if \\( x_1 = y_1 = z_1 = 0 \\), i.e., point \\( P \\) is origin \\( O \\), then \\( OQ = \\sqrt{x_2^2 + y_2^2 + z_2^2} \\), which gives the distance between the origin \\( O \\) and any point \\( Q(x_2, y_2, z_2) \\)."
    #         ],
    #         "diagrams": []
    #     },
    #     "Section Formula": {
    #         "content": [
    #             "Let the two given points be \\( P(x_1, y_1, z_1) \\) and \\( Q(x_2, y_2, z_2) \\). Let point \\( R(x, y, z) \\) divide \\( PQ \\) in the given ratio \\( m : n \\) internally. Draw \\( PL \\), \\( QM \\) and \\( RN \\) perpendicular to the \\( XY \\)-plane. Obviously \\( PL \\parallel RN \\parallel QM \\) and the feet",
    #             "---",
    #             "## Page 14",
    #             "of these perpendiculars lie in the XY-plane. Through point R draw a line ST parallel to line LM. Line ST will intersect line LP externally at point S and line MQ at T, as shown in Fig. 1.4.",
    #             "Also note that quadrilaterals LNRS and NMTR are parallelograms. The triangles PSR and QTR are similar. Therefore,",
    #             "\\[",
    #             "\\frac{m}{n} = \\frac{PR}{QR} = \\frac{SP}{QT} = \\frac{SL - PL}{QM - TM} = \\frac{NR - PL}{QM - NR} = \\frac{z - z_1}{z_2 - z}",
    #             "\\]",
    #             "\\[",
    #             "\\Rightarrow \\quad z = \\frac{mz_2 + nz_1}{m + n}",
    #             "\\]",
    #             "Hence, the coordinates of the point R which divides the line segment joining two points \\( P(x_1, y_1, z_1) \\) and \\( Q(x_2, y_2, z_2) \\) internally in the ratio \\( m : n \\) are",
    #             "\\[",
    #             "\\frac{mx_2 + nx_1}{m + n}, \\frac{my_2 + ny_1}{m + n}, \\frac{mz_2 + nz_1}{m + n}",
    #             "\\]",
    #             "If point R divides PQ externally in the ratio \\( m : n \\), then its coordinates are obtained by replacing \\( n \\) with \\( -n \\) so that the coordinates become",
    #             "\\[",
    #             "\\frac{mx_2 - nx_1}{m - n}, \\frac{my_2 - ny_1}{m - n}, \\frac{mz_2 - nz_1}{m - n}",
    #             "\\]",
    #             "**Notes:**",
    #             "1. If R is the midpoint of PQ, then \\( m : n = 1 : 1 \\); so \\( x = \\frac{x_1 + x_2}{2}, y = \\frac{y_1 + y_2}{2}, z = \\frac{z_1 + z_2}{2} \\). These are the coordinates of the midpoint of the segment joining \\( P(x_1, y_1, z_1) \\) and \\( Q(x_2, y_2, z_2) \\).",
    #             "2. The coordinates of the point R which divides PQ in the ratio \\( k : 1 \\) are obtained by taking \\( k = \\frac{m}{n} \\), which are given by \\( \\left( \\frac{kx_2 + x_1}{k + 1}, \\frac{ky_2 + y_1}{k + 1}, \\frac{kz_2 + z_1}{k + 1} \\right) \\).",
    #             "3. If vertices of triangle are \\( A(x_1, y_1, z_1), B(x_2, y_2, z_2) \\) and \\( C(x_3, y_3, z_3) \\), and \\( AB = c, BC = a, AC = b \\), then centroid of the triangle is \\( \\left( \\frac{x_1 + x_2 + x_3}{3}, \\frac{y_1 + y_2 + y_3}{3}, \\frac{z_1 + z_2 + z_3}{3} \\right) \\) and its incenter is \\( \\left( \\frac{ax_1 + bx_2 + cx_3}{a + b + c}, \\frac{ay_1 + by_2 + cy_3}{a + b + c}, \\frac{az_1 + bz_2 + cz_3}{a + b + c} \\right) \\).",
    #             "## EVOLUTION OF VECTOR CONCEPT",
    #             "In our day-to-day life, we come across many queries such as \"What is your height?\" and \"How should a football player hit the ball to give a pass to another player of his team?\" Observe that a possible answer to the first query may be 1.5 m, a quantity that involves only one value (magnitude) which is a real number. Such quantities are called scalars. However, an answer to the second query is a quantity (called force) which involves muscular strength (magnitude) and direction (in which another player is positioned). Such quantities are called vectors. In mathematics, physics and engineering, we frequently come across with both types of quantities, namely scalar quantities such as length, mass, time, distance, speed, area, volume, temperature, work, money, voltage, density and resistance and vector quantities such as displacement, velocity, acceleration, force, momentum and electric field intensity.",
    #             "Let 'l' be a straight line in a plane or a three-dimensional space. This line can be given two directions by means of arrowheads. A line with one of these directions prescribed is called a directed line [Fig.1.5 (i), (ii)].",
    #             "---",
    #             "## Page 15"
    #         ],
    #         "diagrams": [
    #             "The diagram shows a 3D coordinate system with axes labeled \\( X \\), \\( Y \\), and \\( Z \\). It illustrates a rectangular parallelepiped formed by planes through points \\( P \\) and \\( Q \\). The points \\( P \\) and \\( Q \\) are connected by a diagonal \\( PQ \\). The diagram includes perpendiculars \\( PL \\), \\( QM \\), and \\( RN \\) drawn from points \\( P \\), \\( Q \\), and \\( R \\) respectively to the \\( XY \\)-plane.",
    #             "\nThe diagram shows a three-dimensional coordinate system with axes labeled X, Y, and Z. It includes points P, Q, R, S, T, L, M, and N. The diagram illustrates the geometric relationships and intersections described in the text, particularly focusing on the line segments and parallelograms formed by these points.\n"
    #         ]
    #     },
    #     "1.4 Vectors and 3D Geometry": {
    #         "content": [
    #             "Now observe that if we restrict the line *l* to the line segment *AB*, then a magnitude is prescribed on line (i) with one of the two directions, so that we obtain a directed line segment, Fig. 1.5 (iii). Thus, a directed line segment has magnitude as well as direction.",
    #             "**Definition**",
    #             "A quantity that has magnitude as well as direction is called a vector.",
    #             "Notice that a directed line segment is a vector [Fig. 1.5(iii)], denoted as $\\overrightarrow{AB}$ or simply as $\\vec{a}$, and read as \"vector $\\overrightarrow{AB}$\" or \"vector $\\vec{a}$\".",
    #             "Point *A* from where vector $\\overrightarrow{AB}$ starts is called its initial point, and point *B* where it ends is called its terminal point. The distance between initial and terminal points of a vector is called the magnitude (or length) of the vector, denoted as $|\\overrightarrow{AB}|$ or $|\\vec{a}|$ or $a$. The arrow indicates the direction of the vector.",
    #             "**Position Vector**",
    #             "Consider a point *P* in space having coordinates $(x, y, z)$ with respect to the origin *O* $(0, 0, 0)$. Then, the vector $\\overrightarrow{OP}$ having *O* and *P* as its initial and terminal points, respectively, is called the position vector of the point *P* with respect to *O*. Using the distance formula, the magnitude of $\\overrightarrow{OP}$ (or $\\vec{r}$) is given by $|\\overrightarrow{OP}| = \\sqrt{x^2 + y^2 + z^2}$.",
    #             "In practice, the position vectors of points *A*, *B*, *C*, etc., with respect to origin *O* are denoted by $\\vec{a}$, $\\vec{b}$, $\\vec{c}$, etc., respectively [Fig. 1.6(ii)].",
    #             "**Direction Cosines**",
    #             "Consider the position vector $\\overrightarrow{OP}$ (or $\\vec{r}$) of a point *P* $(x, y, z)$. The angles $\\alpha$, $\\beta$, and $\\gamma$ made by the vector $\\vec{r}$ with the positive directions of *x*-, *y*-, and *z*-axes, respectively, are called its direction angles. The cosine values of these angles, i.e., $\\cos \\alpha$, $\\cos \\beta$, and $\\cos \\gamma$, are called direction cosines of the vector $\\vec{r}$ and are usually denoted by *l*, *m*, and *n*, respectively.",
    #             "---",
    #             "## Page 16",
    #             "From Fig. 1.7, one may note that triangle $OAP$ is right angled, and in it, we have $\\cos \\alpha = x/r$ ($r$ stands for $|\\vec{r}|$). Similarly, from the right-angled triangles $ORP$ and $OCP$, we may write $\\cos \\beta = y/r$ and $\\cos \\gamma = z/r$. Thus, the coordinates of point $P$ may also be expressed as $(lr, mr, nr)$. The numbers $l$, $m$ and $n$, proportional to the direction cosines, are called the direction ratios of vector $\\vec{r}$ and are denoted by $a$, $b$ and $c$, respectively (see this topic in detail in Chapter 3).",
    #             "## TYPES OF VECTORS"
    #         ],
    #         "diagrams": [
    #             "\nFig. 1.6 (i):\n- The diagram shows a 3D coordinate system with axes labeled *x*, *y*, and *z*.\n- A point *P* with coordinates $(x, y, z)$ is shown.\n- The origin *O* is at $(0, 0, 0)$.\n- A vector $\\overrightarrow{OP}$ is drawn from the origin *O* to the point *P*.\n\nFig. 1.6 (ii):\n- The diagram shows a 3D coordinate system with axes labeled *x*, *y*, and *z*.\n- Points *A*, *B*, and *C* are shown with position vectors $\\vec{a}$, $\\vec{b}$, and $\\vec{c}$ respectively.\n- Vectors $\\overrightarrow{OA}$, $\\overrightarrow{OB}$, and $\\overrightarrow{OC}$ are drawn from the origin *O* to points *A*, *B*, and *C* respectively.\n"
    #         ]
    #     },
    #     "Zero Vector": {
    #         "content": [
    #             "A vector whose initial and terminal points coincide is called a zero vector (or null vector) and is denoted as $\\vec{0}$. A zero vector cannot be assigned a definite direction as it has zero magnitude or, alternatively, it may be regarded as having any direction. The vectors $\\overrightarrow{AA}$, $\\overrightarrow{BB}$ represent the zero vector."
    #         ],
    #         "diagrams": []
    #     },
    #     "Unit Vector": {
    #         "content": [
    #             "A vector of unit magnitude is called a unit vector. Unit vectors are denoted by small letters with a cap on them. Thus, $\\hat{a}$ is unit vector of $\\vec{a}$, where $|\\hat{a}| = 1$, i.e., if vector $\\vec{a}$ is divided by magnitude $|\\vec{a}|$, then we get a unit vector in the direction of $\\vec{a}$. Thus, $\\hat{a} = \\frac{\\vec{a}}{|\\vec{a}|} \\Leftrightarrow \\vec{a} = |\\vec{a}|\\hat{a}$, where $\\hat{a}$ is the unit vector in the direction of $\\vec{a}$."
    #         ],
    #         "diagrams": []
    #     },
    #     "Coinitial Vectors": {
    #         "content": [
    #             "Two or more vectors having the same initial point are called coinitial vectors."
    #         ],
    #         "diagrams": []
    #     },
    #     "Equal Vectors": {
    #         "content": [
    #             "Two vectors $\\vec{a}$ and $\\vec{b}$ are said to be equal if they have the same magnitude and direction regardless of the positions of their initial points. They are written as $\\vec{a} = \\vec{b}$."
    #         ],
    #         "diagrams": []
    #     },
    #     "Negative of a Vector": {
    #         "content": [
    #             "A vector whose magnitude is the same as that of a given vector (say, $\\overrightarrow{AB}$), but whose direction is opposite to that of it, is called negative of the given vector. For example, vector $\\overrightarrow{BA}$ is negative of vector $\\overrightarrow{AB}$ and is written as $\\overrightarrow{BA} = -\\overrightarrow{AB}$."
    #         ],
    #         "diagrams": []
    #     },
    #     "Free Vectors": {
    #         "content": [
    #             "Vectors whose initial points are not specified are called free vectors."
    #         ],
    #         "diagrams": []
    #     },
    #     "Localised Vectors": {
    #         "content": [
    #             "A vector drawn parallel to a given vector, but through a specified point as the initial point, is called a localised vector.",
    #             "---",
    #             "## Page 17",
    #             "## 1.6 Vectors and 3D Geometry"
    #         ],
    #         "diagrams": [
    #             "Fig. 1.7 shows a 3D coordinate system with axes labeled $X$, $Y$, and $Z$. The point $P$ has coordinates $(x, y, z)$ and is connected to the origin $O$. The angles $\\alpha$, $\\beta$, and $\\gamma$ are shown between the position vector $\\vec{r}$ and the $X$, $Y$, and $Z$ axes respectively. The diagram also shows the right-angled triangles $OAP$, $ORP$, and $OCP$ with $O$ as the origin and $P$ as the terminal point.",
    #             "Fig. 1.8 shows vectors $\\vec{a}$, $\\vec{b}$, and $\\vec{c}$. The vector $\\vec{a}$ is represented with an arrow pointing in a specific direction. The vector $\\hat{a}$ is shown as a unit vector in the direction of $\\vec{a}$. The vector $\\vec{c}$ is shown as equal to $\\vec{a}$."
    #         ]
    #     },
    #     "Parallel Vectors": {
    #         "content": [
    #             "Two or more vectors are said to be parallel if they have the same support or parallel support.",
    #             "Parallel vectors may have equal or unequal magnitudes and their directions may be same or opposite as shown in Fig. 1.9."
    #         ],
    #         "diagrams": [
    #             "Fig. 1.9 shows three vectors:\n- Vector $\\vec{OA}$ and vector $\\vec{AB}$ are parallel and in the same direction.\n- Vector $\\vec{CD}$ and vector $\\vec{CB}$ are parallel and in opposite directions."
    #         ]
    #     }
    # }
    

    try:
        # Initialize system
        rag = ImprovedRAGSystem()
        
        # Load and prepare documents
        print("Loading and preparing documents...")
        rag.prepare_documents_from_json("vector_content.json")
        
        print("Creating vector store...")
        rag.create_vectorstore()
        
        print("\nSystem ready! Ask questions about vectors and 3D geometry.")
        print("Type 'quit' to exit")
        print("-" * 50)
    
        # Query the system
        while True:
            query = input("\nEnter your question (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
                
            print("\nSearching for relevant information...")
            result = rag.query(query)
            
            print("\nAnswer:", result["answer"])
            print("\nSources used:")
            for source in result["sources"]:
                print(f"- {source}")
            print("-" * 50)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
