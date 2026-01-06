'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import styles from './RolloutDetail.module.css';
import TurnHeader from './TurnHeader';
import ActionDetails from './ActionDetails';

interface RolloutDetailProps {
  rolloutId: number;
  selectedTurnIndex?: number | null;
  onClose: () => void;
  onTurnChange?: (turnIndex: number | null) => void;
}

interface Turn {
  id: number;
  turn: number;
  reward: number;
  episode_done: boolean;
  turn_time: number;
  start_time: string;
  end_time: string;
  model_response: string | null;
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
  selectedTurnIndex: propSelectedTurnIndex,
  onClose,
  onTurnChange,
}: RolloutDetailProps) {
  const [data, setData] = useState<RolloutDetailData | null>(null);
  const [loading, setLoading] = useState(true);
  // Use turns.length as the index for Validation tab, -1 for Env Build tab
  const [selectedTurnIndex, setSelectedTurnIndex] = useState<number>(propSelectedTurnIndex !== null && propSelectedTurnIndex !== undefined ? propSelectedTurnIndex : -1);
  const [showScreenshotModal, setShowScreenshotModal] = useState(false);
  const [modalImageSrc, setModalImageSrc] = useState<string | null>(null);
  const [showModelInputModal, setShowModelInputModal] = useState(false);
  const [modelInputData, setModelInputData] = useState<any>(null);
  const [loadingModelInput, setLoadingModelInput] = useState(false);
  // Cache for loaded turn details, screenshots and model inputs
  const [turnDetailsCache, setTurnDetailsCache] = useState<Map<number, { actions: any[], observations: any[] }>>(new Map());
  const [loadingTurnDetails, setLoadingTurnDetails] = useState<Set<number>>(new Set());
  const [screenshotCache, setScreenshotCache] = useState<Map<number, string>>(new Map());
  const [modelInputCache, setModelInputCache] = useState<Map<number, any>>(new Map());
  const [loadingScreenshots, setLoadingScreenshots] = useState<Set<number>>(new Set());
  const [showVideoModal, setShowVideoModal] = useState(false);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  
  // Use refs to access latest cache values in callbacks without dependencies
  const turnDetailsCacheRef = useRef<Map<number, { actions: any[], observations: any[] }>>(new Map());
  const modelInputCacheRef = useRef<Map<number, any>>(new Map());
  const loadingTurnDetailsRef = useRef<Set<number>>(new Set());
  const screenshotCacheRef = useRef<Map<number, string>>(new Map());
  
  // Keep refs in sync with state - update refs directly without useEffect to avoid loops
  // We'll update refs in the callbacks that modify state instead
  
  const beforeScreenshotRef = useRef<HTMLImageElement | null>(null);
  const beforeCanvasRef = useRef<HTMLCanvasElement | null>(null);
  
  // Calculate selectedTurnId using ref to track previous value and avoid loops
  // MUST be before any conditional returns (React hooks rule)
  const prevSelectedTurnIdRef = useRef<number | null>(null);

  const fetchDetails = async () => {
    try {
      const res = await fetch(`/api/rollouts/${rolloutId}`);
      const data = await res.json();
      console.log('Fetched rollout data:', data);
      console.log('Rollout object:', data.rollout);
      console.log('Rollout keys:', data.rollout ? Object.keys(data.rollout) : 'no rollout');
      console.log('trajectory_data_json exists:', !!data.rollout?.trajectory_data_json);
      console.log('trajectory_data_json type:', typeof data.rollout?.trajectory_data_json);
      if (data.rollout?.trajectory_data_json) {
        try {
          const parsed = typeof data.rollout.trajectory_data_json === 'string' 
            ? JSON.parse(data.rollout.trajectory_data_json)
            : data.rollout.trajectory_data_json;
          console.log('Parsed trajectory_data_json:', parsed);
          console.log('Has turns:', !!parsed.turns);
          if (parsed.turns) {
            console.log('Turns count:', parsed.turns.length);
            console.log('First turn:', parsed.turns[0]);
          }
        } catch (e) {
          console.error('Failed to parse trajectory_data_json:', e);
        }
      }
      setData(data);
    } catch (error) {
      console.error('Failed to fetch rollout details:', error);
    } finally {
      setLoading(false);
    }
  };

  // Define ALL callbacks and functions BEFORE any conditional returns
  // This is required by React - hooks must be called in the same order every render
  
  const handleScreenshotClick = useCallback((imageSrc: string) => {
    setModalImageSrc(imageSrc);
    setShowScreenshotModal(true);
  }, []);

  const handleCloseModal = useCallback(() => {
    setShowScreenshotModal(false);
    setModalImageSrc(null);
  }, []);

  const handleOpenModelInputModal = useCallback(async (turnId: number) => {
    // Check cache first using ref
    const cached = modelInputCacheRef.current.get(turnId);
    if (cached) {
      setModelInputData(cached);
      setShowModelInputModal(true);
      return;
    }

    // Load from API
    setLoadingModelInput(true);
    try {
      const res = await fetch(`/api/rollouts/${rolloutId}/model-input/${turnId}`);
      if (!res.ok) {
        throw new Error('Failed to load model input');
      }
      const data = await res.json();
      const modelInput = data.modelInput;
      const screenshotUri = data.screenshot_uri || null;
      const payload = (modelInput && typeof modelInput === 'object')
        ? { ...modelInput, screenshot_uri: screenshotUri }
        : { modelInput, screenshot_uri: screenshotUri };
      
      // Cache it using functional update
      setModelInputCache(prev => new Map(prev).set(turnId, payload));
      setModelInputData(payload);
      setShowModelInputModal(true);
    } catch (error) {
      console.error('Failed to load model input:', error);
      alert('Failed to load model input');
    } finally {
      setLoadingModelInput(false);
    }
  }, [rolloutId]);

  const handleCloseModelInputModal = useCallback(() => {
    setShowModelInputModal(false);
    setModelInputData(null);
  }, []);

  const handleOpenVideoModal = useCallback(() => {
    // Construct video URL from trajectory_path
    if (data?.rollout?.trajectory_path) {
      const trajectoryPath = data.rollout.trajectory_path;
      // trajectory_path is like "logs/trajectories/step_1_batch_0_group_0_rollout_abc123"
      // Video file is at trajectory_path/recording.mp4
      const videoPath = `${trajectoryPath}/recording.mp4`;
      // Use API route to serve video
      const apiUrl = `/api/videos?path=${encodeURIComponent(videoPath)}`;
      setVideoUrl(apiUrl);
      setShowVideoModal(true);
    } else {
      alert('No video available for this rollout');
    }
  }, [data?.rollout?.trajectory_path]);

  const handleCloseVideoModal = useCallback(() => {
    setShowVideoModal(false);
    setVideoUrl(null);
  }, []);

  // Function to load turn details on demand
  const loadTurnDetails = useCallback(async (turnId: number) => {
    // Load from API (cache check is done in useEffect)
    setLoadingTurnDetails(prev => new Set(prev).add(turnId));
    try {
      const res = await fetch(`/api/rollouts/${rolloutId}/turns/${turnId}`);
      if (!res.ok) {
        throw new Error('Failed to load turn details');
      }
      const data = await res.json();
      
      // Cache it using functional update
      setTurnDetailsCache(prev => new Map(prev).set(turnId, data));
      return data;
    } catch (error) {
      console.error('Failed to load turn details:', error);
      return null;
    } finally {
      setLoadingTurnDetails(prev => {
        const next = new Set(prev);
        next.delete(turnId);
        return next;
      });
    }
  }, [rolloutId]);

  // Function to load screenshot URI for an observation
  // Now screenshot_uri is a file path, so we can use it directly or load from static files
  const loadScreenshot = useCallback(async (obsId: number, screenshotUri?: string | null): Promise<string | null> => {
    // If we have a file path, use it directly (it's already a static file path)
    if (screenshotUri && !screenshotUri.startsWith('data:') && !screenshotUri.startsWith('http')) {
      // It's a file path, return it as-is (will be served as static file)
      return screenshotUri;
    }
    
    // Check cache first using ref
    const cached = screenshotCacheRef.current.get(obsId);
    if (cached) {
      return cached;
    }

    // If it's a data URI or URL, cache it
    if (screenshotUri) {
      setScreenshotCache(prev => new Map(prev).set(obsId, screenshotUri));
      return screenshotUri;
    }

    // Fallback: try to load from API (for old data)
    setLoadingScreenshots(prev => new Set(prev).add(obsId));
    try {
      const res = await fetch(`/api/rollouts/${rolloutId}/screenshot/${obsId}`);
      if (!res.ok) {
        throw new Error('Failed to load screenshot');
      }
      const data = await res.json();
      const uri = data.screenshot_uri;
      
      // Cache it
      if (uri) {
        setScreenshotCache(prev => new Map(prev).set(obsId, uri));
      }
      return uri || null;
    } catch (error) {
      console.error('Failed to load screenshot:', error);
      return null;
    } finally {
      setLoadingScreenshots(prev => {
        const next = new Set(prev);
        next.delete(obsId);
        return next;
      });
    }
  }, [rolloutId]);

  // All useEffect hooks must be before conditional returns
  useEffect(() => {
    fetchDetails();
    // Auto-refresh every 2 seconds
    const interval = setInterval(fetchDetails, 2000);
    return () => clearInterval(interval);
  }, [rolloutId]);

  // Sync propSelectedTurnIndex to state
  useEffect(() => {
    if (propSelectedTurnIndex !== null && propSelectedTurnIndex !== undefined) {
      setSelectedTurnIndex(propSelectedTurnIndex);
    }
  }, [propSelectedTurnIndex]);

  // Validate selectedTurnIndex when turns data changes
  // Use a stable reference to turns length
  const turnsLength = data?.turns?.length ?? 0;
  useEffect(() => {
    if (turnsLength > 0) {
      // Allow -1 for Env Build, 0 to turns.length-1 for turns, turns.length for Validation
      setSelectedTurnIndex((currentIndex) => {
        // If currentIndex is out of bounds (beyond turns.length), reset to -1 (Env Build)
        if (currentIndex > turnsLength) {
          return -1;
        }
        return currentIndex;
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [turnsLength]);

  // Helper function to extract turn data from trajectory_data_json
  const extractTurnData = useCallback((rollout: any, turn: any) => {
    if (!rollout || !turn) {
      console.log('extractTurnData: Missing rollout or turn', { rollout: !!rollout, turn: !!turn });
      return null;
    }
    
    if (!rollout.trajectory_data_json) {
      console.log('extractTurnData: No trajectory_data_json in rollout', { rolloutKeys: Object.keys(rollout || {}) });
      return null;
    }
    
    try {
      const trajectoryData = typeof rollout.trajectory_data_json === 'string'
        ? JSON.parse(rollout.trajectory_data_json)
        : rollout.trajectory_data_json;
      
      console.log('extractTurnData: Parsed trajectoryData', { 
        hasExecutionDetails: !!trajectoryData.execution_details,
        hasTurns: !!trajectoryData.execution_details?.turns,
        turnsLength: trajectoryData.execution_details?.turns?.length,
        lookingForTurn: turn.turn,
        trajectoryDataKeys: Object.keys(trajectoryData || {})
      });
      
      // New structure: { training_data: [...], execution_details: { turns: [...] } }
      // Old structure (backward compatibility): { turns: [...] }
      const executionDetails = trajectoryData.execution_details || trajectoryData;
      
      if (executionDetails.turns && Array.isArray(executionDetails.turns)) {
        const turnData = executionDetails.turns.find((t: any) => t.turn_num === turn.turn);
        console.log('extractTurnData: Found turnData', { 
          found: !!turnData, 
          turnDataKeys: turnData ? Object.keys(turnData) : null,
          hasActionResults: turnData?.action_results?.length > 0,
          hasToolExecutions: turnData?.tool_executions?.length > 0
        });
        return turnData || null;
      }
    } catch (e) {
      console.error('extractTurnData: Parse error', e);
    }
    
    return null;
  }, []);

  // Draw coordinates on screenshot
  const drawCoordinatesOnScreenshot = useCallback((img: HTMLImageElement, canvas: HTMLCanvasElement, coordinates: any) => {
    if (!img || !canvas || !coordinates) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Get displayed image dimensions
    const displayedWidth = img.offsetWidth || img.clientWidth;
    const displayedHeight = img.offsetHeight || img.clientHeight;
    const naturalWidth = img.naturalWidth || img.width;
    const naturalHeight = img.naturalHeight || img.height;
    
    // Set canvas size to match displayed image
    canvas.width = displayedWidth;
    canvas.height = displayedHeight;
    
    // Calculate scale factors
    const scaleX = displayedWidth / naturalWidth;
    const scaleY = displayedHeight / naturalHeight;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw coordinates
    if (coordinates.x !== undefined && coordinates.y !== undefined) {
      // Single point (tap, click, etc.)
      const x = coordinates.x * scaleX;
      const y = coordinates.y * scaleY;
      
      // Draw outer circle (larger, more visible)
      ctx.strokeStyle = '#ff0000';
      ctx.fillStyle = 'rgba(255, 0, 0, 0.2)';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.arc(x, y, Math.max(20, 30 * scaleX), 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
      
      // Draw inner circle
      ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
      ctx.beginPath();
      ctx.arc(x, y, Math.max(8, 12 * scaleX), 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw crosshair (longer, more visible)
      ctx.strokeStyle = '#ff0000';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(x - 30 * scaleX, y);
      ctx.lineTo(x + 30 * scaleX, y);
      ctx.moveTo(x, y - 30 * scaleY);
      ctx.lineTo(x, y + 30 * scaleY);
      ctx.stroke();
      
      // Draw label with background
      const labelText = `(${coordinates.x}, ${coordinates.y})`;
      const fontSize = Math.max(12, 16 * scaleX);
      ctx.font = `bold ${fontSize}px Arial`;
      const textMetrics = ctx.measureText(labelText);
      const textWidth = textMetrics.width;
      const textHeight = fontSize;
      const padding = 4;
      const labelX = x + 35 * scaleX;
      const labelY = y - 15 * scaleY;
      
      // Draw label background
      ctx.fillStyle = 'rgba(255, 0, 0, 0.9)';
      ctx.fillRect(labelX - padding, labelY - textHeight, textWidth + padding * 2, textHeight + padding * 2);
      
      // Draw label text
      ctx.fillStyle = '#ffffff';
      ctx.fillText(labelText, labelX, labelY);
    } else if (coordinates.start && coordinates.end) {
      // Two points (swipe, drag, etc.)
      const startX = coordinates.start.x * scaleX;
      const startY = coordinates.start.y * scaleY;
      const endX = coordinates.end.x * scaleX;
      const endY = coordinates.end.y * scaleY;
      
      // Draw arrow line between points
      ctx.strokeStyle = '#0066ff';
      ctx.lineWidth = 4;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(startX, startY);
      ctx.lineTo(endX, endY);
      ctx.stroke();
      
      // Draw arrowhead at end point
      const angle = Math.atan2(endY - startY, endX - startX);
      const arrowLength = 20 * scaleX;
      ctx.beginPath();
      ctx.moveTo(endX, endY);
      ctx.lineTo(
        endX - arrowLength * Math.cos(angle - Math.PI / 6),
        endY - arrowLength * Math.sin(angle - Math.PI / 6)
      );
      ctx.moveTo(endX, endY);
      ctx.lineTo(
        endX - arrowLength * Math.cos(angle + Math.PI / 6),
        endY - arrowLength * Math.sin(angle + Math.PI / 6)
      );
      ctx.stroke();
      
      // Draw start point (green)
      ctx.strokeStyle = '#00cc00';
      ctx.fillStyle = 'rgba(0, 204, 0, 0.3)';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.arc(startX, startY, Math.max(15, 20 * scaleX), 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = 'rgba(0, 204, 0, 0.6)';
      ctx.beginPath();
      ctx.arc(startX, startY, Math.max(6, 10 * scaleX), 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw end point (red)
      ctx.strokeStyle = '#ff0000';
      ctx.fillStyle = 'rgba(255, 0, 0, 0.3)';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.arc(endX, endY, Math.max(15, 20 * scaleX), 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = 'rgba(255, 0, 0, 0.6)';
      ctx.beginPath();
      ctx.arc(endX, endY, Math.max(6, 10 * scaleX), 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw start label with background
      const startText = `Start (${coordinates.start.x}, ${coordinates.start.y})`;
      const fontSize = Math.max(12, 16 * scaleX);
      ctx.font = `bold ${fontSize}px Arial`;
      const startMetrics = ctx.measureText(startText);
      const padding = 4;
      const startLabelX = startX + 25 * scaleX;
      const startLabelY = startY - 10 * scaleY;
      
      ctx.fillStyle = 'rgba(0, 204, 0, 0.9)';
      ctx.fillRect(startLabelX - padding, startLabelY - fontSize, startMetrics.width + padding * 2, fontSize + padding * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(startText, startLabelX, startLabelY);
      
      // Draw end label with background
      const endText = `End (${coordinates.end.x}, ${coordinates.end.y})`;
      const endMetrics = ctx.measureText(endText);
      const endLabelX = endX + 25 * scaleX;
      const endLabelY = endY - 10 * scaleY;
      
      ctx.fillStyle = 'rgba(255, 0, 0, 0.9)';
      ctx.fillRect(endLabelX - padding, endLabelY - fontSize, endMetrics.width + padding * 2, fontSize + padding * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(endText, endLabelX, endLabelY);
    }
  }, []);

  // Calculate selectedTurnId (before any conditional returns or useEffect hooks that use it)
  const selectedTurnId = (() => {
    if (!data?.turns || data.turns.length === 0) return null;
    if (selectedTurnIndex < 0) return null;  // Env Build tab
    if (selectedTurnIndex >= data.turns.length) return null;
    return data.turns[selectedTurnIndex]?.id ?? null;
  })();

  // Effect to draw coordinates when screenshot loads or coordinates change
  useEffect(() => {
    if (!data || !data.turns || data.turns.length === 0) return;
    if (selectedTurnIndex >= data.turns.length) return;
    
    const currentTurn = data.turns[selectedTurnIndex];
    if (!currentTurn) return;
    
    // Get coordinates from action_results in trajectory_data_json
    const turnData = extractTurnData(data.rollout, currentTurn);
    const actionResults = turnData?.action_results || [];
    const actionResult = actionResults.length > 0 ? actionResults[0] : null;
    const coordinates = actionResult?.coordinates;
    
    console.log('[Coordinates Debug]', {
      turnNum: currentTurn.turn,
      hasTurnData: !!turnData,
      hasActionResults: actionResults.length > 0,
      hasCoordinates: !!coordinates,
      coordinates: coordinates,
      actionResult: actionResult,
      hasRefs: !!(beforeScreenshotRef.current && beforeCanvasRef.current)
    });
    
    // Use a small delay to ensure refs are updated after render
    const timeoutId = setTimeout(() => {
      if (beforeScreenshotRef.current && beforeCanvasRef.current) {
        if (coordinates) {
          const img = beforeScreenshotRef.current;
          const canvas = beforeCanvasRef.current;
          
          const drawWhenReady = () => {
            // Use requestAnimationFrame to ensure layout is complete
            requestAnimationFrame(() => {
              drawCoordinatesOnScreenshot(img, canvas, coordinates);
            });
          };
          
          // Always set onload handler (in case image loads after this effect runs)
          img.onload = drawWhenReady;
          
          // If image is already loaded, draw immediately
          if (img.complete && img.naturalWidth > 0) {
            drawWhenReady();
          } else {
            // If image is not loaded yet, wait for it
            img.onload = drawWhenReady;
          }
        } else {
          // Clear canvas if no coordinates
          const ctx = beforeCanvasRef.current.getContext('2d');
          if (ctx) {
            ctx.clearRect(0, 0, beforeCanvasRef.current.width, beforeCanvasRef.current.height);
          }
        }
      }
    }, 50); // Small delay to ensure DOM is updated
    
    return () => clearTimeout(timeoutId);
    // extractTurnData and drawCoordinatesOnScreenshot are stable (empty deps), so we can omit them
    // Also depend on turnDetailsCache to redraw when turn details are loaded
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTurnIndex, data?.rollout?.id, data?.turns?.length, turnDetailsCache, selectedTurnId]);
  
  // Also redraw when window resizes
  useEffect(() => {
    const handleResize = () => {
      if (!data || !data.turns || data.turns.length === 0) return;
      if (selectedTurnIndex >= data.turns.length) return;
      
      const currentTurn = data.turns[selectedTurnIndex];
      if (!currentTurn) return;
      
      if (beforeScreenshotRef.current && beforeCanvasRef.current) {
        // Get coordinates from action_results in trajectory_data_json
        const turnData = extractTurnData(data.rollout, currentTurn);
        const actionResults = turnData?.action_results || [];
        const actionResult = actionResults.length > 0 ? actionResults[0] : null;
        const coordinates = actionResult?.coordinates;
        
        if (coordinates) {
          const img = beforeScreenshotRef.current;
          const canvas = beforeCanvasRef.current;
          drawCoordinatesOnScreenshot(img, canvas, coordinates);
        }
      }
    };
    
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [selectedTurnIndex, data, extractTurnData, drawCoordinatesOnScreenshot]);
  
  // Sync refs with state (only for reading, not as dependencies)
  turnDetailsCacheRef.current = turnDetailsCache;
  loadingTurnDetailsRef.current = loadingTurnDetails;
  screenshotCacheRef.current = screenshotCache;
  modelInputCacheRef.current = modelInputCache;
  
  // Load turn details when selectedTurnId actually changes (not just on every render)
  useEffect(() => {
    // Only load if the turn ID actually changed
    if (selectedTurnId === prevSelectedTurnIdRef.current) {
      return;
    }
    
    // Update the ref
    prevSelectedTurnIdRef.current = selectedTurnId;
    
    if (!selectedTurnId || !data?.turns) return;
    
    // Check if already cached using ref (not state to avoid dependency)
    if (turnDetailsCacheRef.current.has(selectedTurnId)) {
      // Even if current turn is cached, we should preload next turn for after screenshot
      const currentTurnIndex = data.turns.findIndex((t: any) => t && t.id === selectedTurnId);
      if (currentTurnIndex >= 0 && currentTurnIndex < data.turns.length - 1) {
        const nextTurn = data.turns[currentTurnIndex + 1];
        if (nextTurn && !turnDetailsCacheRef.current.has(nextTurn.id) && !loadingTurnDetailsRef.current.has(nextTurn.id)) {
          loadTurnDetails(nextTurn.id);
        }
      }
      return;
    }
    
    // Check if already loading using ref
    if (loadingTurnDetailsRef.current.has(selectedTurnId)) {
      return;
    }
    
    // Load the current turn details
    loadTurnDetails(selectedTurnId);
    
    // Also preload next turn's details (if exists) for after screenshot
    const currentTurnIndex = data.turns.findIndex((t: any) => t && t.id === selectedTurnId);
    if (currentTurnIndex >= 0 && currentTurnIndex < data.turns.length - 1) {
      const nextTurn = data.turns[currentTurnIndex + 1];
      if (nextTurn && !turnDetailsCacheRef.current.has(nextTurn.id) && !loadingTurnDetailsRef.current.has(nextTurn.id)) {
        loadTurnDetails(nextTurn.id);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTurnId, data?.turns]);
  
  // NOW we can have conditional returns (all hooks are above)
  if (loading) {
    return <div className={styles.loading}>Loading...</div>;
  }

  if (!data) {
    return <div className={styles.error}>Rollout not found</div>;
  }

  const { rollout, task, validation, environment, turns = [] } = data || {};
  
  // Safety check: ensure rollout exists
  if (!rollout) {
    return <div className={styles.error}>Rollout data is invalid</div>;
  }
  
  // Get selectedTurn directly from data
  const selectedTurn = selectedTurnId && data?.turns 
    ? data.turns.find((t: any) => t?.id === selectedTurnId) || null
    : null;
  
  // Get turn details from cache (loaded on demand)
  const selectedTurnDetails = selectedTurnId ? turnDetailsCache.get(selectedTurnId) : null;
  const selectedTurnActions = selectedTurnDetails?.actions || [];
  const selectedTurnObservations = selectedTurnDetails?.observations || [];
  const isLoadingTurnDetails = selectedTurnId ? loadingTurnDetails.has(selectedTurnId) : false;

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div style={{ flex: 1 }}>
          <h2 className={styles.title}>Rollout Details</h2>
          {task && (
            <div style={{ 
              marginTop: '6px',
              fontSize: '12px',
              color: '#666',
              display: 'flex',
              gap: '8px',
              alignItems: 'center'
            }}>
              <span style={{ fontWeight: 600, color: '#333' }}>{task.name}</span>
              <span style={{ color: '#999' }}>•</span>
              <span style={{ 
                fontFamily: 'monospace',
                fontSize: '11px',
                backgroundColor: '#f0f0f0',
                padding: '2px 6px',
                borderRadius: '3px',
                color: '#666'
              }}>
                ID: {task.id}
              </span>
            </div>
          )}
        </div>
        <div className={styles.headerButtons}>
          {rollout?.trajectory_path && (
            <button 
              className={styles.refreshButton}
              onClick={handleOpenVideoModal}
              title="Play Recording"
              style={{ marginRight: '8px' }}
            >
              🎥
            </button>
          )}
          <button 
            className={styles.refreshButton} 
            onClick={fetchDetails}
            title="Refresh"
          >
            🔄
          </button>
          <button className={styles.closeButton} onClick={onClose}>
            ← Back
          </button>
        </div>
      </div>

      <div className={styles.content}>
        {/* Compact Summary Row */}
        <div className={styles.summaryRow}>
          <div className={styles.summaryItem}>
            <span className={styles.summaryLabel}>Status:</span>
            <span className={styles.summaryValue}>{rollout.status}</span>
          </div>
          <div className={styles.summaryItem}>
            <span className={styles.summaryLabel}>Task:</span>
            <span className={`${styles.summaryValue} ${rollout.task_success === true ? styles.success : styles.failed}`}>
              {rollout.task_success === true ? '✓' : '✗'}
            </span>
          </div>
          <div className={styles.summaryItem}>
            <span className={styles.summaryLabel}>Validation:</span>
            <span className={`${styles.summaryValue} ${rollout.validation_passed === true ? styles.success : styles.failed}`}>
              {rollout.validation_passed === true ? '✓' : '✗'}
            </span>
          </div>
          <div className={styles.summaryItem}>
            <span className={styles.summaryLabel}>Turns:</span>
            <span className={styles.summaryValue}>{rollout.num_turns || turns.length}</span>
          </div>
          <div className={styles.summaryItem}>
            <span className={styles.summaryLabel}>Reward:</span>
            <span className={styles.summaryValue}>
              {rollout.reward !== null ? rollout.reward.toFixed(4) : 'N/A'}
            </span>
          </div>
          <div className={styles.summaryItem}>
            <span className={styles.summaryLabel}>Time:</span>
            <span className={styles.summaryValue}>
              {rollout.rollout_time !== null ? `${rollout.rollout_time.toFixed(1)}s` : 'N/A'}
            </span>
          </div>
          {task && (
            <div className={styles.summaryItem}>
              <span className={styles.summaryLabel}>Task:</span>
              <span className={styles.summaryValue} title={task.description}>
                {task.name.length > 30 ? task.name.substring(0, 30) + '...' : task.name}
              </span>
            </div>
          )}
          {rollout?.trajectory_path && (
            <div className={styles.summaryItem}>
              <button
                onClick={handleOpenVideoModal}
                style={{
                  padding: '4px 12px',
                  backgroundColor: '#6f42c1',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '12px',
                  fontWeight: 500,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.backgroundColor = '#5a32a3';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.backgroundColor = '#6f42c1';
                }}
              >
                🎥 Play Recording
              </button>
            </div>
          )}
        </div>

        {/* Task and Validation Details (if available) */}
        {(task || validation) && (
          <div className={styles.detailsRow}>
            {task && (
              <div className={styles.detailSection}>
                <span className={styles.detailLabel}>Task Description:</span>
                <span className={styles.detailValue} style={{ 
                  color: '#212529', 
                  fontSize: '13px',
                  fontWeight: 500,
                  lineHeight: '1.5'
                }}>
                  {task.description}
                </span>
              </div>
            )}
            {validation && validation.validation_query && (
              <div className={styles.detailSection}>
                <span className={styles.detailLabel}>Validation:</span>
                <code className={styles.detailCode}>{validation.validation_query}</code>
                {validation.expected_result && (
                  <>
                    <span className={styles.detailLabel}>Expected:</span>
                    <code className={styles.detailCode}>{validation.expected_result}</code>
                  </>
                )}
                {validation.actual_result && (
                  <>
                    <span className={styles.detailLabel}>Actual:</span>
                    <code className={styles.detailCode}>{validation.actual_result}</code>
                  </>
                )}
              </div>
            )}
          </div>
        )}

        {/* Turns Tabs */}
        {turns.length > 0 && (
          <div className={styles.turnsSection}>
            <div className={styles.tabsContainer}>
              <div className={styles.tabs}>
                {/* Env Build Tab */}
                {(() => {
                  // Get env build time
                  let envBuildTime: number | null = null;
                  try {
                    const trajectoryData = typeof rollout.trajectory_data_json === 'string'
                      ? JSON.parse(rollout.trajectory_data_json)
                      : rollout.trajectory_data_json;
                    const executionDetails = trajectoryData?.execution_details || trajectoryData;
                    envBuildTime = executionDetails?.env_build?.total_time;
                  } catch (e) {
                    // Ignore parse errors
                  }
                  
                  return (
                    <button
                      className={`${styles.tab} ${selectedTurnIndex === -1 ? styles.tabActive : ''}`}
                      onClick={() => {
                        setSelectedTurnIndex(-1);
                        if (onTurnChange) {
                          onTurnChange(-1);
                        }
                      }}
                    >
                      <span className={styles.tabNumber}>Env Build</span>
                      {envBuildTime !== null && envBuildTime !== undefined && (
                        <span className={styles.tabBadge} style={{ 
                          backgroundColor: '#6c757d',
                          marginLeft: '4px',
                          fontSize: '11px',
                          padding: '2px 6px'
                        }}>
                          {envBuildTime.toFixed(1)}s
                        </span>
                      )}
                    </button>
                  );
                })()}
                {turns.map((turn, index) => (
                  <button
                    key={turn.id}
                    className={`${styles.tab} ${selectedTurnIndex === index ? styles.tabActive : ''}`}
                    onClick={() => {
                      setSelectedTurnIndex(index);
                      if (onTurnChange) {
                        onTurnChange(index);
                      }
                    }}
                  >
                    <span className={styles.tabNumber}>Turn {turn.turn}</span>
                    {turn.turn_time !== null && turn.turn_time !== undefined && (
                      <span className={styles.tabBadge} style={{ 
                        backgroundColor: '#6c757d',
                        marginLeft: '4px',
                        fontSize: '11px',
                        padding: '2px 6px'
                      }}>
                        {turn.turn_time.toFixed(1)}s
                      </span>
                    )}
                    {turn.episode_done && <span className={styles.tabBadge}>Final</span>}
                  </button>
                ))}
                {validation && (
                  <button
                    className={`${styles.tab} ${selectedTurnIndex === turns.length ? styles.tabActive : ''}`}
                    onClick={() => {
                      setSelectedTurnIndex(turns.length);
                      if (onTurnChange) {
                        onTurnChange(turns.length);
                      }
                    }}
                  >
                    <span className={styles.tabNumber}>Validation</span>
                    {validation.execution_time !== null && validation.execution_time !== undefined && (
                      <span className={styles.tabBadge} style={{ 
                        backgroundColor: '#6c757d',
                        marginLeft: '4px',
                        fontSize: '11px',
                        padding: '2px 6px'
                      }}>
                        {validation.execution_time.toFixed(1)}s
                      </span>
                    )}
                  </button>
                )}
              </div>
            </div>

            {/* Selected Turn Content or Validation Content */}
            {selectedTurnIndex === -1 ? (
              /* Env Build Content */
              <div className={styles.turnContent}>
                {(() => {
                  // Parse env_build from trajectory_data_json
                  let envBuildData: any = null;
                  try {
                    const trajectoryData = typeof rollout.trajectory_data_json === 'string'
                      ? JSON.parse(rollout.trajectory_data_json)
                      : rollout.trajectory_data_json;
                    
                    // Check for env_build in execution_details (new structure) or top-level (old structure)
                    const executionDetails = trajectoryData?.execution_details || trajectoryData;
                    envBuildData = executionDetails?.env_build;
                  } catch (e) {
                    console.error('Failed to parse env_build data:', e);
                  }
                  
                  if (!envBuildData || !envBuildData.stages || envBuildData.stages.length === 0) {
                    return (
                      <div style={{ padding: '20px', textAlign: 'center', color: '#999' }}>
                        No environment build information available
                      </div>
                    );
                  }
                  
                  const { stages, total_time, status, box_id, box_type, apk_path, apk_size_mb, prehook_executed, prehook_output } = envBuildData;
                  
                  return (
                    <div style={{ padding: '20px' }}>
                      {/* Summary */}
                      <div style={{ 
                        marginBottom: '24px', 
                        padding: '16px', 
                        backgroundColor: status === 'success' ? '#d4edda' : status === 'error' ? '#f8d7da' : '#fff3cd',
                        borderRadius: '8px',
                        border: `1px solid ${status === 'success' ? '#c3e6cb' : status === 'error' ? '#f5c6cb' : '#ffeaa7'}`
                      }}>
                        <h3 style={{ margin: '0 0 12px 0', fontSize: '18px', fontWeight: 600 }}>
                          Environment Build Summary
                        </h3>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', fontSize: '14px' }}>
                          <div>
                            <strong>Status:</strong>{' '}
                            <span style={{ 
                              color: status === 'success' ? '#155724' : status === 'error' ? '#721c24' : '#856404',
                              fontWeight: 600 
                            }}>
                              {status === 'success' ? '✓ Success' : status === 'error' ? '✗ Failed' : '⋯ In Progress'}
                            </span>
                          </div>
                          <div>
                            <strong>Total Time:</strong> {total_time ? `${total_time.toFixed(3)}s` : 'N/A'}
                          </div>
                          {box_id && (
                            <div>
                              <strong>Box ID:</strong> <code>{box_id}</code>
                            </div>
                          )}
                          {box_type && (
                            <div>
                              <strong>Box Type:</strong> {box_type}
                            </div>
                          )}
                          {apk_size_mb && (
                            <div>
                              <strong>APK Size:</strong> {apk_size_mb} MB
                            </div>
                          )}
                          <div>
                            <strong>Prehook:</strong> {prehook_executed ? '✓ Executed' : '✗ Not Executed'}
                          </div>
                        </div>
                      </div>
                      
                      {/* Timeline */}
                      <div style={{ marginBottom: '24px' }}>
                        <h3 style={{ margin: '0 0 16px 0', fontSize: '18px', fontWeight: 600 }}>
                          Build Timeline
                        </h3>
                        <div style={{ position: 'relative', paddingLeft: '40px' }}>
                          {/* Vertical line */}
                          <div style={{
                            position: 'absolute',
                            left: '16px',
                            top: '8px',
                            bottom: '8px',
                            width: '2px',
                            backgroundColor: '#dee2e6'
                          }} />
                          
                          {stages.map((stage: any, index: number) => {
                            const stageStatus = stage.status;
                            const stageDuration = stage.duration;
                            const stageDetails = stage.details || {};
                            const stageError = stage.error;
                            
                            const statusColor = stageStatus === 'success' ? '#28a745' : 
                                              stageStatus === 'error' ? '#dc3545' : '#ffc107';
                            const statusIcon = stageStatus === 'success' ? '✓' : 
                                             stageStatus === 'error' ? '✗' : '⋯';
                            
                            return (
                              <div key={index} style={{ 
                                position: 'relative', 
                                marginBottom: '24px',
                                paddingLeft: '8px'
                              }}>
                                {/* Timeline dot */}
                                <div style={{
                                  position: 'absolute',
                                  left: '-32px',
                                  top: '4px',
                                  width: '16px',
                                  height: '16px',
                                  borderRadius: '50%',
                                  backgroundColor: statusColor,
                                  border: '3px solid white',
                                  boxShadow: '0 0 0 2px ' + statusColor,
                                  display: 'flex',
                                  alignItems: 'center',
                                  justifyContent: 'center',
                                  fontSize: '10px',
                                  color: 'white',
                                  fontWeight: 'bold'
                                }} />
                                
                                <div style={{
                                  backgroundColor: 'white',
                                  border: '1px solid #dee2e6',
                                  borderRadius: '8px',
                                  padding: '12px 16px',
                                  boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
                                }}>
                                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                                    <div style={{ fontSize: '16px', fontWeight: 600 }}>
                                      {statusIcon} {stage.name}
                                    </div>
                                    {stageDuration !== null && stageDuration !== undefined && (
                                      <div style={{ 
                                        fontSize: '14px', 
                                        color: '#666',
                                        backgroundColor: '#f8f9fa',
                                        padding: '4px 8px',
                                        borderRadius: '4px'
                                      }}>
                                        {stageDuration.toFixed(3)}s
                                      </div>
                                    )}
                                  </div>
                                  
                                  {/* Stage Details */}
                                  {Object.keys(stageDetails).length > 0 && (
                                    <div style={{ 
                                      fontSize: '13px', 
                                      color: '#666',
                                      marginTop: '8px',
                                      paddingTop: '8px',
                                      borderTop: '1px solid #f0f0f0'
                                    }}>
                                      {Object.entries(stageDetails).map(([key, value]) => (
                                        <div key={key} style={{ marginBottom: '4px' }}>
                                          <strong>{key}:</strong> {String(value)}
                                        </div>
                                      ))}
                                    </div>
                                  )}
                                  
                                  {/* Stage Error */}
                                  {stageError && (
                                    <div style={{ 
                                      marginTop: '8px',
                                      padding: '8px 12px',
                                      backgroundColor: '#f8d7da',
                                      border: '1px solid #f5c6cb',
                                      borderRadius: '4px',
                                      color: '#721c24',
                                      fontSize: '13px'
                                    }}>
                                      <strong>Error:</strong> {stageError}
                                    </div>
                                  )}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                      
                      {/* Prehook Output */}
                      {prehook_output && (
                        <div style={{ marginBottom: '24px' }}>
                          <h3 style={{ margin: '0 0 12px 0', fontSize: '18px', fontWeight: 600 }}>
                            Prehook Output
                          </h3>
                          <pre style={{
                            backgroundColor: '#f8f9fa',
                            border: '1px solid #dee2e6',
                            borderRadius: '4px',
                            padding: '12px',
                            fontSize: '13px',
                            lineHeight: '1.5',
                            overflow: 'auto',
                            maxHeight: '300px',
                            whiteSpace: 'pre-wrap',
                            wordBreak: 'break-word'
                          }}>
                            {prehook_output}
                          </pre>
                        </div>
                      )}
                    </div>
                  );
                })()}
              </div>
            ) : selectedTurnIndex < turns.length && selectedTurn ? (
              <div className={styles.turnContent}>
                {/* Compact Turn Header with Integrated Timeline */}
                {(() => {
                  // Re-fetch turn details for this scope (TypeScript scoping issue)
                  const turnDetails = selectedTurnId ? turnDetailsCache.get(selectedTurnId) : null;
                  const turnActions = turnDetails?.actions || [];
                  const action = turnActions && turnActions.length > 0 ? turnActions[0] : null;
                  const turnData = extractTurnData(rollout, selectedTurn);
                  
                  // Parse metrics_json to get precise stage timings
                  let stageTimings: any = {};
                  try {
                    const metricsJson = (selectedTurn as any).metrics_json;
                    if (metricsJson) {
                      const metrics = typeof metricsJson === 'string' ? JSON.parse(metricsJson) : metricsJson;
                      stageTimings = metrics.stage_timings || {};
                    }
                  } catch (e) {
                    console.error('Failed to parse metrics_json:', e);
                  }
                  
                  return (
                    <TurnHeader
                      turn={selectedTurn}
                      action={action}
                      stageTimings={stageTimings}
                      parseSuccess={turnData?.parse_success}
                      loadingModelInput={loadingModelInput}
                      onOpenModelInput={() => {
                        if (selectedTurn) {
                          handleOpenModelInputModal(selectedTurn.id);
                        }
                      }}
                    />
                  );
                })()}

                {/* OLD Action Details Section REMOVED - Will be added after Model Response */}

                {/* Turn Content: Left (Screenshots) and Right (Model Response) */}
                <div className={styles.turnContentLayout}>
                  {/* Left: Screenshots (Before and After) */}
                  <div className={styles.turnLeftPanel}>
                    {(() => {
                      if (isLoadingTurnDetails) {
                        return (
                          <div className={styles.turnSection}>
                            <h4 className={styles.turnSectionTitle}>Screenshots</h4>
                            <div className={styles.emptyScreenshot}>Loading turn details...</div>
                          </div>
                        );
                      }
                      
                      if (!selectedTurn || !selectedTurnObservations || !Array.isArray(selectedTurnObservations)) {
                        return (
                          <div className={styles.turnSection}>
                            <h4 className={styles.turnSectionTitle}>Screenshots</h4>
                            <div className={styles.emptyScreenshot}>No screenshots available</div>
                          </div>
                        );
                      }
                      
                      // Find Before screenshot observation (from current turn)
                      const beforeObs = selectedTurnObservations.find(
                        (obs: any) => obs && obs.obs_type === 'screenshot_before'
                      );
                      
                      // Find After screenshot (from next turn's Before screenshot, or from Validation if this is the last turn)
                      const currentTurnIndex = turns.findIndex((t: any) => t && t.id === selectedTurn.id);
                      const nextTurn = currentTurnIndex >= 0 && currentTurnIndex < turns.length - 1 ? turns[currentTurnIndex + 1] : null;
                      const isLastTurn = currentTurnIndex >= 0 && currentTurnIndex === turns.length - 1;
                      
                      // For last turn, use Validation screenshot if available
                      let afterObs = null;
                      let afterObsId: number | null = null;
                      if (isLastTurn && validation) {
                        // Try to get screenshot from validation details_json
                        try {
                          const details = typeof validation.details_json === 'string' 
                            ? JSON.parse(validation.details_json) 
                            : validation.details_json;
                          if (details && details.screenshot_uri) {
                            afterObs = { screenshot_uri: details.screenshot_uri };
                          }
                        } catch (e) {
                          // Ignore parse errors
                        }
                      }
                      
                      // If not last turn or no validation screenshot, use next turn's Before screenshot
                      if (!afterObs && nextTurn) {
                        const nextTurnDetails = turnDetailsCache.get(nextTurn.id);
                        afterObs = nextTurnDetails?.observations?.find(
                          (obs: any) => obs && obs.obs_type === 'screenshot_before'
                        );
                        if (afterObs) {
                          afterObsId = afterObs.id;
                        }
                      }
                      
                      if (!beforeObs) {
                        return (
                          <div className={styles.turnSection}>
                            <h4 className={styles.turnSectionTitle}>Screenshots</h4>
                            <div className={styles.emptyScreenshot}>No screenshots available</div>
                          </div>
                        );
                      }
                      
                      // Get screenshot URIs (now they're file paths, not base64)
                      const beforeUri = beforeObs?.screenshot_uri || null;
                      const afterUri = afterObs?.screenshot_uri || null;
                      
                      // If screenshot_uri is a file path (not data URI or URL), prepend static path
                      const getScreenshotUrl = (uri: string | null) => {
                        if (!uri) return null;
                        // If it's already a data URI or full URL, return as-is
                        if (uri.startsWith('data:') || uri.startsWith('http://') || uri.startsWith('https://')) {
                          return uri;
                        }
                        // Otherwise, it's a file path - serve via API route
                        // Remove leading slash if present, then prepend /api/screenshots/
                        const cleanPath = uri.startsWith('/') ? uri.slice(1) : uri;
                        // Remove 'screenshots/' prefix if present (to avoid double prefix)
                        const normalizedPath = cleanPath.startsWith('screenshots/') ? cleanPath.slice('screenshots/'.length) : cleanPath;
                        return `/api/screenshots/${normalizedPath}`;
                      };
                      
                      const beforeUrl = getScreenshotUrl(beforeUri);
                      const afterUrl = getScreenshotUrl(afterUri);
                      
                      // Check if we have coordinates to show
                      const turnData = extractTurnData(rollout, selectedTurn);
                      const actionResults = turnData?.action_results || [];
                      const actionResult = actionResults.length > 0 ? actionResults[0] : null;
                      const coordinates = actionResult?.coordinates;
                      const hasCoordinates = coordinates !== null;
                      
                      const renderScreenshot = (url: string | null, label: string, alt: string, isLoading: boolean, showCoordinates: boolean = false) => {
                        const isBefore = label === 'Before';
                        
                        if (isLoading) {
                          return (
                            <div style={{ flex: '1', minWidth: 0 }}>
                              <h5 style={{ marginBottom: '10px', fontSize: '14px', fontWeight: 600, color: '#666' }}>
                                {label}
                              </h5>
                              <div className={styles.emptyScreenshot} style={{ maxHeight: '600px' }}>
                                Loading screenshot...
                              </div>
                            </div>
                          );
                        }
                        
                        if (!url) {
                          return (
                            <div style={{ flex: '1', minWidth: 0 }}>
                              <h5 style={{ marginBottom: '10px', fontSize: '14px', fontWeight: 600, color: '#666' }}>
                                {label}
                              </h5>
                              <div className={styles.emptyScreenshot} style={{ maxHeight: '600px' }}>
                                No screenshot available
                              </div>
                            </div>
                          );
                        }
                        
                        return (
                          <div style={{ flex: '1', minWidth: 0 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
                              <h5 style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: '#666' }}>
                                {label}
                              </h5>
                            </div>
                            <div 
                              className={styles.screenshotContainer}
                              onClick={() => handleScreenshotClick(url)}
                              style={{ cursor: 'pointer', position: 'relative' }}
                            >
                              <div style={{ position: 'relative', display: 'inline-block', maxWidth: '100%' }}>
                                <img
                                  ref={isBefore ? beforeScreenshotRef : null}
                                  src={url}
                                  alt={alt}
                                  className={styles.screenshotImg}
                                  style={{
                                    maxWidth: '100%',
                                    maxHeight: '600px',
                                    width: 'auto',
                                    height: 'auto',
                                    objectFit: 'contain',
                                    display: 'block',
                                  }}
                                  onLoad={() => {
                                    // Trigger coordinates redraw when image loads
                                    if (isBefore && showCoordinates && beforeScreenshotRef.current && beforeCanvasRef.current) {
                                      // Use closure to access current values
                                      const currentTurnIndex = selectedTurnIndex;
                                      const currentData = data;
                                      if (currentData?.turns && currentTurnIndex < currentData.turns.length) {
                                        const currentTurn = currentData.turns[currentTurnIndex];
                                        if (currentTurn && currentData.rollout) {
                                          const turnData = extractTurnData(currentData.rollout, currentTurn);
                                          const actionResults = turnData?.action_results || [];
                                          const actionResult = actionResults.length > 0 ? actionResults[0] : null;
                                          const coordinates = actionResult?.coordinates;
                                          if (coordinates) {
                                            requestAnimationFrame(() => {
                                              if (beforeScreenshotRef.current && beforeCanvasRef.current) {
                                                drawCoordinatesOnScreenshot(
                                                  beforeScreenshotRef.current,
                                                  beforeCanvasRef.current,
                                                  coordinates
                                                );
                                              }
                                            });
                                          }
                                        }
                                      }
                                    }
                                  }}
                                />
                                {isBefore && showCoordinates && (
                                  <canvas
                                    ref={beforeCanvasRef}
                                    style={{
                                      position: 'absolute',
                                      top: 0,
                                      left: 0,
                                      pointerEvents: 'none',
                                      maxWidth: '100%',
                                      maxHeight: '600px',
                                    }}
                                  />
                                )}
                              </div>
                            </div>
                          </div>
                        );
                      };
                      
                      return (
                        <div className={styles.turnSection}>
                          <h4 className={styles.turnSectionTitle}>Screenshots</h4>
                          <div style={{ display: 'flex', gap: '20px', alignItems: 'flex-start' }}>
                            {renderScreenshot(
                              beforeUrl,
                              'Before',
                              `Turn ${selectedTurn.turn} Before screenshot`,
                              false,
                              hasCoordinates
                            )}
                            {afterObs || afterObsId ? (
                              renderScreenshot(
                                afterUrl,
                                'After',
                                `Turn ${selectedTurn.turn} After screenshot`,
                                false,
                                false
                              )
                            ) : (
                              <div style={{ flex: '1', minWidth: 0 }}>
                                <h5 style={{ marginBottom: '10px', fontSize: '14px', fontWeight: 600, color: '#666' }}>
                                  After
                                </h5>
                                <div className={styles.emptyScreenshot} style={{ maxHeight: '600px' }}>
                                  {isLastTurn ? 'No After screenshot available' : 'No After screenshot (this is the last turn)'}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      );
                    })()}
                  </div>

                  {/* Right: Model Response */}
                  <div className={styles.turnRightPanel}>
                    {isLoadingTurnDetails ? (
                      <div className={styles.turnSection}>
                        <h4 className={styles.turnSectionTitle}>Loading turn details...</h4>
                      </div>
                    ) : (
                      <>
                        {/* Model Response */}
                        {selectedTurn.model_response && (
                          <div className={styles.turnSection}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
                              <h4 className={styles.turnSectionTitle} style={{ marginBottom: 0 }}>Model Response</h4>
                              {(() => {
                                const turnDetails = selectedTurnId ? turnDetailsCache.get(selectedTurnId) : null;
                                const turnActions = turnDetails?.actions || [];
                                const action = turnActions && turnActions.length > 0 ? turnActions[0] : null;
                                if (action && action.num_tokens !== null && action.num_tokens !== undefined) {
                                  return (
                                    <span style={{ 
                                      fontSize: '11px', 
                                      color: '#666',
                                      fontWeight: 600,
                                      backgroundColor: '#f0f0f0',
                                      padding: '2px 8px',
                                      borderRadius: '3px',
                                      border: '1px solid #dee2e6'
                                    }}>
                                      {action.num_tokens} tokens
                                    </span>
                                  );
                                }
                                return null;
                              })()}
                            </div>
                            <div className={styles.modelResponseContainer}>
                              <pre className={styles.modelResponseText}>{selectedTurn.model_response}</pre>
                            </div>
                          </div>
                        )}
                        
                        {/* Action Details - Placed after Model Response for compact layout */}
                        {(() => {
                          const turnData = extractTurnData(rollout, selectedTurn);
                          const actionResults = turnData?.action_results || [];
                          const actionResult = actionResults.length > 0 ? actionResults[0] : undefined;
                          
                          return (
                            <ActionDetails
                              parseSuccess={turnData?.parse_success}
                              parseError={turnData?.parse_error}
                              actionResult={actionResult}
                            />
                          );
                        })()}
                      </>
                    )}
                  </div>
                </div>
              </div>
            ) : selectedTurnIndex === turns.length && validation ? (
              /* Validation Content */
              <div className={styles.turnContent}>
                {/* Validation Status */}
                <div style={{ marginBottom: '20px' }}>
                  <div style={{ 
                    display: 'inline-block', 
                    padding: '4px 12px', 
                    borderRadius: '4px',
                    backgroundColor: validation.success ? '#d4edda' : '#f8d7da',
                    color: validation.success ? '#155724' : '#721c24',
                    fontWeight: 600,
                    fontSize: '14px'
                  }}>
                    {validation.success ? '✓ Passed' : '✗ Failed'}
                  </div>
                </div>
                
                {/* Validation Details */}
                <div style={{ marginBottom: '20px' }}>
                  {validation.validation_query && (
                    <div style={{ marginBottom: '10px' }}>
                      <strong>Query:</strong> <code>{validation.validation_query}</code>
                    </div>
                  )}
                  {validation.expected_result && (
                    <div style={{ marginBottom: '10px' }}>
                      <strong>Expected:</strong> <code>{validation.expected_result}</code>
                    </div>
                  )}
                  {validation.actual_result && (
                    <div style={{ marginBottom: '10px' }}>
                      <strong>Actual:</strong> <code>{validation.actual_result}</code>
                    </div>
                  )}
                  {validation.execution_time !== null && validation.execution_time !== undefined && (
                    <div style={{ marginBottom: '10px' }}>
                      <strong>Execution Time:</strong> {validation.execution_time.toFixed(3)}s
                    </div>
                  )}
                  {validation.error_message && (
                    <div style={{ marginBottom: '10px', color: '#dc3545' }}>
                      <strong>Error:</strong> {validation.error_message}
                    </div>
                  )}
                </div>

                {/* Validation Screenshot */}
                {(() => {
                  try {
                    const details = typeof validation.details_json === 'string' 
                      ? JSON.parse(validation.details_json) 
                      : validation.details_json;
                    let screenshotUri = details?.screenshot_uri;
                    
                    if (screenshotUri) {
                      // Convert file path to URL if needed
                      if (!screenshotUri.startsWith('data:') && !screenshotUri.startsWith('http://') && !screenshotUri.startsWith('https://')) {
                        const cleanPath = screenshotUri.startsWith('/') ? screenshotUri.slice(1) : screenshotUri;
                        // Remove 'screenshots/' prefix if present (to avoid double prefix)
                        const normalizedPath = cleanPath.startsWith('screenshots/') ? cleanPath.slice('screenshots/'.length) : cleanPath;
                        screenshotUri = `/api/screenshots/${normalizedPath}`;
                      }
                      
                      return (
                        <div>
                          <h5 style={{ marginBottom: '10px', fontSize: '14px', fontWeight: 600, color: '#666' }}>
                            Screenshot
                          </h5>
                          <div 
                            className={styles.screenshotContainer}
                            onClick={() => handleScreenshotClick(screenshotUri)}
                            style={{ cursor: 'pointer' }}
                          >
                            <img
                              src={screenshotUri}
                              alt="Validation screenshot"
                              className={styles.screenshotImg}
                              style={{
                                maxWidth: '100%',
                                maxHeight: '600px',
                                width: 'auto',
                                height: 'auto',
                                objectFit: 'contain',
                              }}
                            />
                          </div>
                        </div>
                      );
                    }
                  } catch (e) {
                    // Ignore parse errors
                  }
                  return null;
                })()}
              </div>
            ) : null}

          </div>
        )}
      </div>

      {/* Screenshot Modal */}
      {showScreenshotModal && modalImageSrc && (
        <div 
          className={styles.modalOverlay}
          onClick={handleCloseModal}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.9)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
            cursor: 'pointer',
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              maxWidth: '95vw',
              maxHeight: '95vh',
              position: 'relative',
            }}
          >
            <button
              onClick={handleCloseModal}
              style={{
                position: 'absolute',
                top: '-40px',
                right: '0',
                background: 'rgba(255, 255, 255, 0.2)',
                border: 'none',
                color: 'white',
                fontSize: '24px',
                width: '32px',
                height: '32px',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              ×
            </button>
            <img
              src={modalImageSrc}
              alt="Full screenshot"
              style={{
                maxWidth: '95vw',
                maxHeight: '95vh',
                width: 'auto',
                height: 'auto',
                objectFit: 'contain',
              }}
            />
          </div>
        </div>
      )}

      {/* Video Modal */}
      {showVideoModal && videoUrl && (
        <div 
          onClick={handleCloseVideoModal}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.9)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
            cursor: 'pointer',
            padding: '40px',
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              backgroundColor: '#1a1a1a',
              borderRadius: '12px',
              padding: '24px',
              maxWidth: '90vw',
              maxHeight: '90vh',
              width: '100%',
              position: 'relative',
              display: 'flex',
              flexDirection: 'column',
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h3 style={{ margin: 0, fontSize: '24px', fontWeight: 600, color: 'white' }}>
                🎥 Recording
              </h3>
              <button
                onClick={handleCloseVideoModal}
                style={{
                  background: 'rgba(255, 255, 255, 0.1)',
                  border: 'none',
                  color: 'white',
                  fontSize: '28px',
                  width: '40px',
                  height: '40px',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  lineHeight: '1',
                  padding: 0,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  transition: 'background-color 0.2s',
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.2)';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
                }}
              >
                ×
              </button>
            </div>
            <div 
              style={{
                flex: 1,
                overflow: 'auto',
                backgroundColor: '#000',
                borderRadius: '8px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <video
                controls
                autoPlay
                style={{
                  maxWidth: '100%',
                  maxHeight: '100%',
                  width: 'auto',
                  height: 'auto',
                }}
                onError={(e) => {
                  console.error('Video load error:', e);
                  alert('Failed to load video. The video file may not exist or is not accessible.');
                }}
              >
                <source src={videoUrl} type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            </div>
            <div style={{ marginTop: '16px', color: '#999', fontSize: '12px', textAlign: 'center' }}>
              {videoUrl}
            </div>
          </div>
        </div>
      )}

      {/* Model Input Modal */}
      {showModelInputModal && modelInputData && (
        <div 
          onClick={handleCloseModelInputModal}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
            cursor: 'pointer',
            padding: '20px',
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              backgroundColor: 'white',
              borderRadius: '8px',
              padding: '24px',
              maxWidth: '90vw',
              maxHeight: '90vh',
              width: '100%',
              position: 'relative',
              display: 'flex',
              flexDirection: 'column',
              boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h3 style={{ margin: 0, fontSize: '20px', fontWeight: 600, color: '#333' }}>Model Input</h3>
              <button
                onClick={handleCloseModelInputModal}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: '#666',
                  fontSize: '28px',
                  width: '32px',
                  height: '32px',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  lineHeight: '1',
                  padding: 0,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.backgroundColor = '#f0f0f0';
                  e.currentTarget.style.color = '#333';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.backgroundColor = 'transparent';
                  e.currentTarget.style.color = '#666';
                }}
              >
                ×
              </button>
            </div>
            <div 
              style={{
                flex: 1,
                overflow: 'auto',
                backgroundColor: '#f8f9fa',
                borderRadius: '4px',
                padding: '16px',
                border: '1px solid #dee2e6',
              }}
            >
              {(() => {
                // Try to parse ModelInput structure and display as messages
                try {
                  // Check if we have the new format with messages
                  if (modelInputData && typeof modelInputData === 'object' && modelInputData.messages && Array.isArray(modelInputData.messages)) {
                    // New format: has readable messages
                    const messages = modelInputData.messages;
                    
                    return (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                        {messages.map((msg: any, idx: number) => (
                          <div
                            key={idx}
                            style={{
                              backgroundColor: 'white',
                              borderRadius: '8px',
                              padding: '16px',
                              border: '1px solid #e0e0e0',
                              boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
                            }}
                          >
                            <div style={{
                              display: 'flex',
                              alignItems: 'center',
                              marginBottom: '12px',
                              paddingBottom: '8px',
                              borderBottom: '1px solid #e0e0e0',
                            }}>
                              <span style={{
                                fontWeight: 600,
                                color: msg.role === 'user' ? '#0066cc' : msg.role === 'assistant' ? '#00aa00' : msg.role === 'system' ? '#9900cc' : '#666',
                                fontSize: '14px',
                                textTransform: 'capitalize',
                                padding: '4px 12px',
                                backgroundColor: msg.role === 'user' ? '#e6f2ff' : msg.role === 'assistant' ? '#e6ffe6' : msg.role === 'system' ? '#f0e6ff' : '#f5f5f5',
                                borderRadius: '4px',
                              }}>
                                {msg.role}
                              </span>
                            </div>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                              {msg.content && Array.isArray(msg.content) ? (
                                msg.content.map((content: any, contentIdx: number) => {
                                  if (content.type === 'image' && (content.url || content.image || content.image_url)) {
                                    const rawImageUrl = content.url || content.image || content.image_url;
                                    const rawImageUrlStr = String(rawImageUrl);
                                    
                                    // Filter out PIL.Image.Image strings from old data
                                    if (rawImageUrlStr.includes('PIL.Image.Image') || rawImageUrlStr.includes('<PIL.')) {
                                      return (
                                        <div key={contentIdx} style={{ 
                                          marginBottom: '8px',
                                          padding: '12px',
                                          backgroundColor: '#fff3cd',
                                          border: '1px solid #ffc107',
                                          borderRadius: '4px',
                                          fontSize: '12px',
                                          color: '#856404'
                                        }}>
                                          ⚠️ Screenshot not available (old data format). Please re-run training to see screenshots.
                                        </div>
                                      );
                                    }
                                    
                                    const resolveImageUrl = (uri: string | null | undefined) => {
                                      if (!uri) return null;
                                      // Placeholder emitted by trainer for the turn's screenshot_before
                                      if (uri === '__SCREENSHOT_BEFORE__' && (modelInputData as any)?.screenshot_uri) {
                                        const raw = String((modelInputData as any).screenshot_uri);
                                        const cleanPath = raw.startsWith('/') ? raw.slice(1) : raw;
                                        // Remove 'screenshots/' prefix if present (to avoid double prefix)
                                        const normalizedPath = cleanPath.startsWith('screenshots/') ? cleanPath.slice('screenshots/'.length) : cleanPath;
                                        return `/api/screenshots/${normalizedPath}`;
                                      }
                                      // Already absolute
                                      if (uri.startsWith('data:') || uri.startsWith('http://') || uri.startsWith('https://')) return uri;
                                      if (uri.startsWith('/api/screenshots/')) return uri;
                                      const clean = uri.startsWith('/') ? uri.slice(1) : uri;
                                      // Some callers may store "screenshots/..." – normalize
                                      const normalizedPath = clean.startsWith('screenshots/') ? clean.slice('screenshots/'.length) : clean;
                                      return `/api/screenshots/${normalizedPath}`;
                                    };
                                    const imageUrl = resolveImageUrl(rawImageUrlStr) || '';
                                    return (
                                      <div key={contentIdx} style={{ marginBottom: '8px' }}>
                                        <img
                                          src={imageUrl}
                                          alt={`Message ${idx} image ${contentIdx}`}
                                          style={{
                                            maxWidth: '100%',
                                            maxHeight: '400px',
                                            borderRadius: '4px',
                                            border: '1px solid #ddd',
                                            objectFit: 'contain',
                                          }}
                                          onError={(e) => {
                                            console.error('Model input image load error:', imageUrl, e);
                                          }}
                                        />
                                      </div>
                                    );
                                  } else if (content.type === 'text' && content.text) {
                                    return (
                                      <div key={contentIdx} style={{ 
                                        fontSize: '14px', 
                                        lineHeight: '1.7', 
                                        color: '#333',
                                        whiteSpace: 'pre-wrap',
                                        wordBreak: 'break-word',
                                      }}>
                                        {content.text}
                                      </div>
                                    );
                                  } else {
                                    return (
                                      <div key={contentIdx} style={{ fontSize: '12px', color: '#999' }}>
                                        Unknown content: {JSON.stringify(content)}
                                      </div>
                                    );
                                  }
                                })
                              ) : typeof msg.content === 'string' ? (
                                <div style={{ 
                                  fontSize: '14px', 
                                  lineHeight: '1.7', 
                                  color: '#333',
                                  whiteSpace: 'pre-wrap',
                                  wordBreak: 'break-word',
                                }}>
                                  {msg.content}
                                </div>
                              ) : (
                                <div style={{ fontSize: '12px', color: '#999' }}>
                                  No content
                                </div>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    );
                  }
                  
                  // Fallback: try to parse chunks structure (old format)
                  let chunks: any[] = [];
                  if (modelInputData && typeof modelInputData === 'object') {
                    if (modelInputData.model_input && modelInputData.model_input.chunks) {
                      chunks = modelInputData.model_input.chunks;
                    } else if (modelInputData.chunks && Array.isArray(modelInputData.chunks)) {
                      chunks = modelInputData.chunks;
                    } else if (Array.isArray(modelInputData)) {
                      chunks = modelInputData;
                    }
                  }
                  
                  if (chunks.length > 0) {
                    return (
                      <div style={{ color: '#666', fontSize: '13px', padding: '12px', backgroundColor: '#fff3cd', borderRadius: '4px', border: '1px solid #ffc107' }}>
                        ⚠️ ModelInput format detected but messages not available. Showing raw structure.
                        <pre style={{ marginTop: '8px', fontSize: '11px', color: '#333' }}>
                          {JSON.stringify(modelInputData, null, 2)}
                        </pre>
                      </div>
                    );
                  }
                  
                  // Final fallback: JSON display
                  return (
                    <pre
                      style={{
                        margin: 0,
                        fontSize: '13px',
                        lineHeight: '1.6',
                        fontFamily: 'Monaco, "Courier New", monospace',
                        color: '#212529',
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word',
                      }}
                    >
                      {JSON.stringify(modelInputData, null, 2)}
                    </pre>
                  );
                } catch (e) {
                  return (
                    <div style={{ color: '#dc3545', fontSize: '14px' }}>
                      Error parsing ModelInput: {String(e)}
                      <pre style={{ marginTop: '8px', fontSize: '12px', color: '#666' }}>
                        {JSON.stringify(modelInputData, null, 2)}
                      </pre>
                    </div>
                  );
                }
              })()}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
