'use client';

import { useState, useEffect } from 'react';
import styles from './RolloutDetail.module.css';

interface RolloutDetailProps {
  rolloutId: number;
  onClose: () => void;
}

interface Turn {
  id: number;
  turn: number;
  reward: number;
  episode_done: boolean;
  turn_time: number;
  start_time: string;
  end_time: string;
  actions: any[];
  observations: any[];
}

interface RolloutDetailData {
  rollout: any;
  task: any;
  validation: any;
  environment: any;
  turns: Turn[];
}

export default function RolloutDetail({
  rolloutId,
  onClose,
}: RolloutDetailProps) {
  const [data, setData] = useState<RolloutDetailData | null>(null);
  const [loading, setLoading] = useState(true);
  const [expandedTurns, setExpandedTurns] = useState<Set<number>>(new Set());

  const fetchDetails = async () => {
    try {
      const res = await fetch(`/api/rollouts/${rolloutId}`);
      const data = await res.json();
      setData(data);
    } catch (error) {
      console.error('Failed to fetch rollout details:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDetails();
    const interval = setInterval(fetchDetails, 2000);
    return () => clearInterval(interval);
  }, [rolloutId]);

  const toggleTurn = (turnNum: number) => {
    const newExpanded = new Set(expandedTurns);
    if (newExpanded.has(turnNum)) {
      newExpanded.delete(turnNum);
    } else {
      newExpanded.add(turnNum);
    }
    setExpandedTurns(newExpanded);
  };

  if (loading) {
    return <div className={styles.loading}>Loading...</div>;
  }

  if (!data) {
    return <div className={styles.error}>Rollout not found</div>;
  }

  const { rollout, task, validation, environment, turns } = data;

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h2 className={styles.title}>Rollout Details</h2>
        <button className={styles.closeButton} onClick={onClose}>
          ← Back
        </button>
      </div>

      <div className={styles.content}>
        {/* Rollout Summary */}
        <div className={styles.section}>
          <h3 className={styles.sectionTitle}>Summary</h3>
          <div className={styles.summaryGrid}>
            <div className={styles.summaryItem}>
              <span className={styles.label}>Status:</span>
              <span className={styles.value}>{rollout.status}</span>
            </div>
            <div className={styles.summaryItem}>
              <span className={styles.label}>Task Success:</span>
              <span
                className={`${styles.value} ${
                  rollout.task_success ? styles.success : styles.failed
                }`}
              >
                {rollout.task_success ? 'Yes' : 'No'}
              </span>
            </div>
            <div className={styles.summaryItem}>
              <span className={styles.label}>Validation:</span>
              <span
                className={`${styles.value} ${
                  rollout.validation_passed ? styles.success : styles.failed
                }`}
              >
                {rollout.validation_passed ? 'Passed' : 'Failed'}
              </span>
            </div>
            <div className={styles.summaryItem}>
              <span className={styles.label}>Turns:</span>
              <span className={styles.value}>{rollout.num_turns}</span>
            </div>
            <div className={styles.summaryItem}>
              <span className={styles.label}>Reward:</span>
              <span className={styles.value}>
                {rollout.reward !== null ? rollout.reward.toFixed(4) : 'N/A'}
              </span>
            </div>
            <div className={styles.summaryItem}>
              <span className={styles.label}>Rollout Time:</span>
              <span className={styles.value}>
                {rollout.rollout_time !== null
                  ? `${rollout.rollout_time.toFixed(2)}s`
                  : 'N/A'}
              </span>
            </div>
          </div>
        </div>

        {/* Task Info */}
        {task && (
          <div className={styles.section}>
            <h3 className={styles.sectionTitle}>Task</h3>
            <div className={styles.taskInfo}>
              <div className={styles.taskName}>{task.name}</div>
              <div className={styles.taskDescription}>{task.description}</div>
            </div>
          </div>
        )}

        {/* Validation */}
        {validation && (
          <div className={styles.section}>
            <h3 className={styles.sectionTitle}>Validation</h3>
            <div className={styles.validationInfo}>
              <div className={styles.validationResult}>
                <span className={styles.label}>Result:</span>
                <span
                  className={`${styles.value} ${
                    validation.success ? styles.success : styles.failed
                  }`}
                >
                  {validation.success ? 'Passed' : 'Failed'}
                </span>
              </div>
              {validation.validation_query && (
                <div className={styles.validationQuery}>
                  <span className={styles.label}>Query:</span>
                  <code className={styles.code}>{validation.validation_query}</code>
                </div>
              )}
              {validation.expected_result && (
                <div className={styles.validationExpected}>
                  <span className={styles.label}>Expected:</span>
                  <code className={styles.code}>{validation.expected_result}</code>
                </div>
              )}
              {validation.actual_result && (
                <div className={styles.validationActual}>
                  <span className={styles.label}>Actual:</span>
                  <code className={styles.code}>{validation.actual_result}</code>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Turns */}
        <div className={styles.section}>
          <h3 className={styles.sectionTitle}>Turns ({turns.length})</h3>
          <div className={styles.turnsList}>
            {turns.map((turn) => {
              const isExpanded = expandedTurns.has(turn.turn);
              return (
                <div key={turn.id} className={styles.turn}>
                  <div
                    className={styles.turnHeader}
                    onClick={() => toggleTurn(turn.turn)}
                  >
                    <span className={styles.expandIcon}>
                      {isExpanded ? '▼' : '▶'}
                    </span>
                    <span className={styles.turnNumber}>Turn {turn.turn}</span>
                    <div className={styles.turnMetrics}>
                      <span className={styles.turnMetric}>
                        Reward: {turn.reward !== null ? turn.reward.toFixed(4) : 'N/A'}
                      </span>
                      {turn.turn_time !== null && (
                        <span className={styles.turnMetric}>
                          Time: {turn.turn_time.toFixed(2)}s
                        </span>
                      )}
                      {turn.episode_done && (
                        <span className={styles.episodeDone}>Final</span>
                      )}
                    </div>
                  </div>
                  {isExpanded && (
                    <div className={styles.turnDetails}>
                      {/* Actions */}
                      {turn.actions.length > 0 && (
                        <div className={styles.turnSection}>
                          <h4 className={styles.turnSectionTitle}>
                            Actions ({turn.actions.length})
                          </h4>
                          {turn.actions.map((action: any, idx: number) => (
                            <div key={action.id || idx} className={styles.action}>
                              <div className={styles.actionHeader}>
                                <span className={styles.actionType}>
                                  {action.tool_name || action.action_type || 'Action'}
                                </span>
                                {action.num_tokens !== null && (
                                  <span className={styles.tokenCount}>
                                    {action.num_tokens} tokens
                                  </span>
                                )}
                              </div>
                              {action.tool_args && (
                                <div className={styles.actionArgs}>
                                  <code className={styles.code}>
                                    {typeof action.tool_args === 'string'
                                      ? action.tool_args
                                      : JSON.stringify(action.tool_args, null, 2)}
                                  </code>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Observations */}
                      {turn.observations.length > 0 && (
                        <div className={styles.turnSection}>
                          <h4 className={styles.turnSectionTitle}>
                            Observations ({turn.observations.length})
                          </h4>
                          {turn.observations.map((obs: any, idx: number) => (
                            <div key={obs.id || idx} className={styles.observation}>
                              <div className={styles.obsHeader}>
                                <span className={styles.obsType}>
                                  {obs.obs_type || 'Observation'}
                                </span>
                              </div>
                              {obs.screenshot_uri && (
                                <div className={styles.screenshot}>
                                  <img
                                    src={obs.screenshot_uri}
                                    alt={`Turn ${turn.turn} screenshot`}
                                    className={styles.screenshotImg}
                                  />
                                </div>
                              )}
                              {obs.text_content && (
                                <div className={styles.textContent}>
                                  {obs.text_content}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}

                      {turn.actions.length === 0 && turn.observations.length === 0 && (
                        <div className={styles.empty}>No actions or observations</div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

