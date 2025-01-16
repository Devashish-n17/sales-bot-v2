import os, psycopg2
from dotenv import load_dotenv
load_dotenv()

db_params = {
    "dbname" : "presales_db",
    "user" : "presales_db_owner",
    "password" : os.getenv('PASSWORD'),
    "host" : os.getenv('HOST'),
    "port" : os.getenv('PORT'),
    "sslmode" : "require"
}

def store_conversation(user_question, bot_answer):
    """Store a conversation in the database."""
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO conversation_history (user_question, bot_answer) VALUES (%s, %s)",
            (user_question, bot_answer)
        )
        conn.commit()
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error storing conversation: {e}")


def get_last_conversations(limit=10):
    """Retrieve the last conversations from the database."""
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_question, bot_answer, timestamp 
            FROM conversation_history 
            ORDER BY timestamp DESC 
            LIMIT %s
        """, (limit,))
        
        conversations = cursor.fetchall()
        
        cursor.close()
        conn.close()
        return conversations

    except Exception as e:
        print(f"❌ Error retrieving conversations: {e}")
        return []
