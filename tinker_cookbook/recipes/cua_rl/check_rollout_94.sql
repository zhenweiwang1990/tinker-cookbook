-- Check rollout 94 data consistency
-- Run this with: psql -d <database_name> -f check_rollout_94.sql

-- Get rollout 94
SELECT 
    r.id,
    r.rollout_id,
    r.task_id,
    r.status,
    r.source_type,
    r.baseline_id,
    t.task_id as task_identifier,
    LEFT(t.description, 200) as task_description_preview
FROM rollout r
LEFT JOIN task t ON r.task_id = t.id
WHERE r.id = 94;

-- Get first turn of rollout 94
SELECT 
    tr.id as turn_id,
    tr.turn,
    tr.rollout_id,
    tr.model_response,
    LEFT(tr.model_response, 300) as model_response_preview
FROM turn tr
WHERE tr.rollout_id = 94
ORDER BY tr.turn ASC
LIMIT 1;

-- Check if there are any turns with wrong rollout_id (foreign key issue)
SELECT 
    tr.id as turn_id,
    tr.turn,
    tr.rollout_id as turn_rollout_id,
    r.id as actual_rollout_id,
    r.rollout_id as rollout_uuid
FROM turn tr
LEFT JOIN rollout r ON tr.rollout_id = r.id
WHERE r.id = 94
ORDER BY tr.turn ASC;

-- Check all tasks to see which one has "no budget limit" description
SELECT 
    id,
    task_id,
    LEFT(description, 300) as description_preview
FROM task
WHERE description LIKE '%预算%' OR description LIKE '%budget%'
ORDER BY id;

