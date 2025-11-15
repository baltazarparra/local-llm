# Personal Assistant Test Scenarios

This document outlines realistic test scenarios to validate the personal assistant functionality.

## Automated Tests

Run the integration tests:
```bash
cd /home/baltz/ego/ego_proxy
python3 test_assistant.py
```

This tests:
- ✅ Database operations (create, insert, query, search)
- ✅ Embedding generation and similarity
- ✅ Semantic search integration
- ✅ Module imports

## Manual Test Scenarios

### Scenario 1: Daily Work Logging

**Objective**: Test basic conversation storage and metadata extraction

**Steps**:
1. Launch assistant: `./assistant.sh`
2. Enter: "I had a meeting with Sarah this morning. We discussed the authentication feature."
3. Enter: "Need to review John's pull request this afternoon."
4. Enter: "Pair programming with Matthew tomorrow - he was stressed last time."
5. Check stats: `/stats`

**Expected Results**:
- All 3 messages saved to database
- People extracted: Sarah, John, Matthew
- Topics extracted: meeting, authentication, pull request, pair programming
- Sentiments captured appropriately

### Scenario 2: Context Retrieval

**Objective**: Test semantic search and context-aware responses

**Steps**:
1. After Scenario 1, ask: "What do you know about Matthew?"
2. Ask: "Tell me about the authentication feature"
3. Ask: "Who have I talked about today?"

**Expected Results**:
- Assistant retrieves relevant past conversations
- Mentions Matthew was stressed
- References Sarah and authentication discussion
- Lists all mentioned people

### Scenario 3: Search Commands

**Objective**: Test manual search functionality

**Steps**:
1. Run: `/search matthew`
2. Run: `/search authentication`
3. Run: `/people`
4. Run: `/topics`

**Expected Results**:
- Search finds Matthew-related conversations
- Search finds authentication discussions
- People list includes Sarah, John, Matthew
- Topics list includes relevant topics

### Scenario 4: Timeline View

**Objective**: Test chronological history

**Steps**:
1. Run: `/timeline`
2. Run: `/timeline Matthew`

**Expected Results**:
- Timeline shows all conversations chronologically
- Filtered timeline shows only Matthew-related conversations
- Metadata displayed (people, topics)

### Scenario 5: Temporal Queries

**Objective**: Test time-based filtering

**Steps**:
1. Add various messages over time
2. Ask: "What did we discuss today?"
3. Ask: "What happened last week?"
4. Ask: "Recent conversations about the project"

**Expected Results**:
- Assistant filters by time appropriately
- Recent conversations prioritized
- Natural language time parsing works

### Scenario 6: Sentiment Tracking

**Objective**: Test emotion and sentiment extraction

**Steps**:
1. Enter: "Great news! The project launch was a success!"
2. Enter: "I'm worried about the deadline for the API project"
3. Enter: "Frustrated with the deployment issues today"
4. Search for each and check metadata

**Expected Results**:
- Positive sentiment for success message
- Concerned/worried sentiment for deadline
- Frustrated/negative sentiment for deployment

### Scenario 7: Complex Queries

**Objective**: Test multi-faceted retrieval

**Steps**:
1. Build up 20-30 conversations about various topics
2. Ask: "What advice do you have for my meeting with Sarah?"
3. Ask: "What should I prioritize this week?"
4. Ask: "Tell me about recent project discussions"

**Expected Results**:
- Assistant combines multiple relevant conversations
- Provides context-aware advice
- References specific past interactions

### Scenario 8: Session Management

**Objective**: Test session persistence

**Steps**:
1. Start assistant, add several messages
2. Run: `/export test_session.md`
3. Exit: `/exit`
4. Launch again: `./assistant.sh`
5. Run: `/stats`
6. Search for previous conversations

**Expected Results**:
- Session exported successfully
- Stats show previous conversations
- Previous messages still searchable
- New session continues from saved state

### Scenario 9: Error Handling

**Objective**: Test robustness

**Steps**:
1. Enter very long message (>1000 words)
2. Enter message with special characters: `@#$%^&*()`
3. Enter empty message
4. Run invalid command: `/invalid`
5. Interrupt generation with Ctrl+C

**Expected Results**:
- Long messages handled gracefully
- Special characters saved correctly
- Empty messages skipped
- Invalid commands show helpful error
- Interruption handled, can continue chatting

### Scenario 10: Performance

**Objective**: Test with larger dataset

**Steps**:
1. Add 50+ conversations
2. Run semantic searches
3. Check response times
4. Run `/stats` to see database size

**Expected Results**:
- Search remains fast (<100ms)
- Database size reasonable (~100-200KB)
- No memory leaks
- Smooth UI experience

## Validation Checklist

After running scenarios, verify:

- [ ] All conversations saved to `assistant_memory.db`
- [ ] Metadata extracted correctly (people, topics, sentiment)
- [ ] Semantic search finds relevant conversations
- [ ] Context injected into responses appropriately
- [ ] Commands work as expected
- [ ] UI renders properly (markdown, tables, panels)
- [ ] No crashes or errors in normal usage
- [ ] Database can be reopened after closing
- [ ] Performance acceptable on CPU
- [ ] Memory usage reasonable

## Performance Benchmarks

Expected performance on typical hardware (CPU):

| Operation | Expected Time |
|-----------|--------------|
| Save message | <10ms |
| Generate embedding | 50-100ms |
| Extract metadata | 2-3 seconds |
| Semantic search | <100ms |
| Full response cycle | 3-5 seconds |
| Database query | <10ms |

## Known Limitations

1. **Metadata extraction is slow** (~2-3 seconds per message)
   - Acceptable for async background processing
   - Doesn't block user interaction

2. **First run downloads embedding model** (~80MB)
   - One-time download
   - Subsequent runs are fast

3. **Extraction accuracy depends on LLM**
   - Smaller models may miss some entities
   - Consider using larger model for better extraction

4. **No concurrent database access**
   - Only one assistant instance at a time
   - SQLite limitation for our use case

## Success Criteria

The implementation is successful if:

✅ All automated tests pass  
✅ Manual scenarios work as described  
✅ No crashes in normal usage  
✅ Performance acceptable  
✅ Memory usage reasonable  
✅ Documentation accurate  
✅ User experience smooth  

## Next Steps

After validating with these scenarios:

1. Use assistant for real daily work logging
2. Gather feedback on context retrieval accuracy
3. Identify any edge cases or issues
4. Consider optimizations if needed
5. Document any additional patterns or use cases

## Troubleshooting Tests

If tests fail:

1. **Check dependencies**: `pip list | grep sentence-transformers`
2. **Check database**: `ls -lh assistant_memory.db`
3. **Check logs**: Look for ERROR messages in output
4. **Verify imports**: `python3 -c "from src.database import AssistantDatabase"`
5. **Test embedding model**: `python3 test_assistant.py`

## Conclusion

These scenarios comprehensively test the personal assistant system. Run them to validate the implementation before using it for real work.

