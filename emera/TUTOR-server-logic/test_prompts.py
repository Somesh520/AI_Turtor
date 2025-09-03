#!/usr/bin/env python3
"""
Test script to verify that all prompts are working correctly.
This script tests that the prompts can be formatted with the variables they expect.
"""

from prompts import (
    ROUTER_PROMPT,
    GRAMMAR_VOCAB_PROMPT,
    TRANSLATOR_PROMPT,
    CONVERSATIONAL_PROMPT,
    CURRICULUM_TUTOR_PROMPT
)

def test_prompts():
    """Test that all prompts can be formatted with expected variables."""
    
    # Test data that matches what the API actually sends
    test_data = {
        "current_question": "What is the meaning of 'namaste'?",
        "previous_query": "Hello",
        "previous_response": "Namaste! How can I help you learn Sanskrit today?",
        "context": "This is a test lesson about Sanskrit greetings.",
        "lesson_to_teach": 1
    }
    
    print("🧪 Testing all prompts...")
    print("=" * 50)
    
    try:
        # Test Router Prompt
        print("✅ Testing ROUTER_PROMPT...")
        formatted_router = ROUTER_PROMPT.format(**test_data)
        print("   Router prompt formatted successfully")
        
        # Test Grammar Prompt
        print("✅ Testing GRAMMAR_VOCAB_PROMPT...")
        formatted_grammar = GRAMMAR_VOCAB_PROMPT.format(**test_data)
        print("   Grammar prompt formatted successfully")
        
        # Test Translator Prompt
        print("✅ Testing TRANSLATOR_PROMPT...")
        formatted_translator = TRANSLATOR_PROMPT.format(**test_data)
        print("   Translator prompt formatted successfully")
        
        # Test Conversational Prompt
        print("✅ Testing CONVERSATIONAL_PROMPT...")
        formatted_conversational = CONVERSATIONAL_PROMPT.format(**test_data)
        print("   Conversational prompt formatted successfully")
        
        # Test Curriculum Prompt
        print("✅ Testing CURRICULUM_TUTOR_PROMPT...")
        formatted_curriculum = CURRICULUM_TUTOR_PROMPT.format(**test_data)
        print("   Curriculum prompt formatted successfully")
        
        print("\n🎉 All prompts are working correctly!")
        print("The variables match what the API provides.")
        
    except KeyError as e:
        print(f"❌ Error: Missing variable in prompt: {e}")
        print("This means a prompt is expecting a variable that isn't provided.")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_prompts()
    if success:
        print("\n✅ All prompts are ready to use!")
    else:
        print("\n❌ There are still issues with the prompts.")

