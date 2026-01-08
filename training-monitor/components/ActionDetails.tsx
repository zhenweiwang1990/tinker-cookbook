/**
 * ActionDetails - Compact action status and details panel
 * Shows parse success/failure prominently with error messages
 * Displays action details (type, coords, timing) in a compact format
 */

import React from 'react';

interface ActionDetailsProps {
  parseSuccess?: boolean;
  parseError?: string;
  actionResult?: {
    action_type?: string;
    coordinates?: {
      x?: number;
      y?: number;
      start?: { x: number; y: number };
      end?: { x: number; y: number };
    };
    coord_time?: number;
    exec_time?: number;
    text?: string;
    error?: string;
    duration?: string | number;
    distance?: string;
    target?: string;
    destination?: string;
    content?: string;
    replace?: boolean;
    pressEnterAfterType?: boolean;
    buttons?: string[];
    direction?: string;
    location?: string;
    // Generic catch-all for other tool-specific parameters
    [key: string]: any;
  };
}

export const ActionDetails: React.FC<ActionDetailsProps> = ({ 
  parseSuccess, 
  parseError, 
  actionResult 
}) => {
  if (parseSuccess === undefined) {
    return null; // No parse info available
  }
  
  return (
    <div style={{ 
      padding: '8px 12px',
      backgroundColor: parseSuccess ? '#d4edda' : '#f8d7da',
      border: `1px solid ${parseSuccess ? '#c3e6cb' : '#f5c6cb'}`,
      borderRadius: '4px',
      marginTop: '8px',
      marginBottom: '8px'
    }}>
      {/* Parse Status */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
        <span style={{ 
          fontSize: '13px',
          fontWeight: 600,
          color: parseSuccess ? '#155724' : '#721c24'
        }}>
          {parseSuccess ? '✓ Action Parsed' : '✗ Parse Failed'}
        </span>
        
        {/* Error Message */}
        {!parseSuccess && parseError && (
          <span style={{ 
            fontSize: '12px',
            color: '#721c24',
            fontStyle: 'italic',
            flex: 1
          }}>
            {parseError}
          </span>
        )}
      </div>
      
      {/* Action Details (only if parse succeeded and we have action data) */}
      {parseSuccess && actionResult && (
        <div style={{ 
          marginTop: '6px',
          fontSize: '11px',
          color: '#666',
          display: 'flex',
          gap: '10px',
          flexWrap: 'wrap'
        }}>
          {/* Action Type */}
          {actionResult.action_type && (
            <span>
              <strong style={{ color: '#444' }}>Type:</strong>{' '}
              <code style={{ 
                backgroundColor: '#fff',
                padding: '1px 5px',
                borderRadius: '2px',
                border: '1px solid #dee2e6',
                fontSize: '11px'
              }}>
                {actionResult.action_type}
              </code>
            </span>
          )}
          
          {/* Coordinates */}
          {actionResult.coordinates && (
            <>
              {actionResult.coordinates.x !== undefined && actionResult.coordinates.y !== undefined && (
                <span>
                  <strong style={{ color: '#444' }}>
                    {actionResult.coordinates_scaled ? 'Scaled Coords' : 'Coords'}:
                  </strong>{' '}
                  <code style={{ 
                    backgroundColor: '#fff',
                    padding: '1px 5px',
                    borderRadius: '2px',
                    border: '1px solid #dee2e6',
                    fontSize: '11px'
                  }}>
                    ({actionResult.coordinates.x}, {actionResult.coordinates.y})
                  </code>
                </span>
              )}
              {/* Show original coordinates if they exist (coordinate scaling was used) */}
              {actionResult.original_coordinates && actionResult.original_coordinates.x !== undefined && actionResult.original_coordinates.y !== undefined && (
                <span>
                  <strong style={{ color: '#444' }}>Original:</strong>{' '}
                  <code style={{ 
                    backgroundColor: '#fff3cd',
                    padding: '1px 5px',
                    borderRadius: '2px',
                    border: '1px solid #ffeaa7',
                    fontSize: '11px'
                  }}>
                    ({actionResult.original_coordinates.x}, {actionResult.original_coordinates.y})
                  </code>
                  {' '}→{' '}
                  <code style={{ 
                    backgroundColor: '#d4edda',
                    padding: '1px 5px',
                    borderRadius: '2px',
                    border: '1px solid #c3e6cb',
                    fontSize: '11px'
                  }}>
                    ({actionResult.coordinates.x}, {actionResult.coordinates.y})
                  </code>
                </span>
              )}
              {actionResult.coordinates.start && actionResult.coordinates.end && (
                <span>
                  <strong style={{ color: '#444' }}>Drag:</strong>{' '}
                  <code style={{ 
                    backgroundColor: '#fff',
                    padding: '1px 5px',
                    borderRadius: '2px',
                    border: '1px solid #dee2e6',
                    fontSize: '11px'
                  }}>
                    ({actionResult.coordinates.start.x}, {actionResult.coordinates.start.y}) → ({actionResult.coordinates.end.x}, {actionResult.coordinates.end.y})
                  </code>
                </span>
              )}
            </>
          )}
          
          {/* Text input */}
          {actionResult.text && (
            <span>
              <strong style={{ color: '#444' }}>Text:</strong>{' '}
              <code style={{ 
                backgroundColor: '#fff',
                padding: '1px 5px',
                borderRadius: '2px',
                border: '1px solid #dee2e6',
                fontSize: '11px',
                maxWidth: '200px',
                display: 'inline-block',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                verticalAlign: 'middle'
              }}>
                {actionResult.text}
              </code>
            </span>
          )}
          
          {/* Tool-specific parameters */}
          {actionResult.duration && (
            <span>
              <strong style={{ color: '#444' }}>Duration:</strong>{' '}
              <code style={{ 
                backgroundColor: '#fff',
                padding: '1px 5px',
                borderRadius: '2px',
                border: '1px solid #dee2e6',
                fontSize: '11px'
              }}>
                {actionResult.duration}
              </code>
            </span>
          )}
          
          {actionResult.distance && (
            <span>
              <strong style={{ color: '#444' }}>Distance:</strong>{' '}
              <code style={{ 
                backgroundColor: '#fff',
                padding: '1px 5px',
                borderRadius: '2px',
                border: '1px solid #dee2e6',
                fontSize: '11px'
              }}>
                {actionResult.distance}
              </code>
            </span>
          )}
          
          {actionResult.target && (
            <span>
              <strong style={{ color: '#444' }}>Target:</strong>{' '}
              <code style={{ 
                backgroundColor: '#fff',
                padding: '1px 5px',
                borderRadius: '2px',
                border: '1px solid #dee2e6',
                fontSize: '11px',
                maxWidth: '200px',
                display: 'inline-block',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                verticalAlign: 'middle'
              }}>
                {actionResult.target}
              </code>
            </span>
          )}
          
          {actionResult.destination && (
            <span>
              <strong style={{ color: '#444' }}>Destination:</strong>{' '}
              <code style={{ 
                backgroundColor: '#fff',
                padding: '1px 5px',
                borderRadius: '2px',
                border: '1px solid #dee2e6',
                fontSize: '11px',
                maxWidth: '200px',
                display: 'inline-block',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                verticalAlign: 'middle'
              }}>
                {actionResult.destination}
              </code>
            </span>
          )}
          
          {actionResult.content && (
            <span>
              <strong style={{ color: '#444' }}>Content:</strong>{' '}
              <code style={{ 
                backgroundColor: '#fff',
                padding: '1px 5px',
                borderRadius: '2px',
                border: '1px solid #dee2e6',
                fontSize: '11px',
                maxWidth: '200px',
                display: 'inline-block',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                verticalAlign: 'middle'
              }}>
                {actionResult.content}
              </code>
            </span>
          )}
          
          {actionResult.replace !== undefined && (
            <span>
              <strong style={{ color: '#444' }}>Replace:</strong>{' '}
              <code style={{ 
                backgroundColor: '#fff',
                padding: '1px 5px',
                borderRadius: '2px',
                border: '1px solid #dee2e6',
                fontSize: '11px'
              }}>
                {actionResult.replace ? 'true' : 'false'}
              </code>
            </span>
          )}
          
          {actionResult.pressEnterAfterType !== undefined && (
            <span>
              <strong style={{ color: '#444' }}>Press Enter:</strong>{' '}
              <code style={{ 
                backgroundColor: '#fff',
                padding: '1px 5px',
                borderRadius: '2px',
                border: '1px solid #dee2e6',
                fontSize: '11px'
              }}>
                {actionResult.pressEnterAfterType ? 'true' : 'false'}
              </code>
            </span>
          )}
          
          {actionResult.buttons && Array.isArray(actionResult.buttons) && (
            <span>
              <strong style={{ color: '#444' }}>Buttons:</strong>{' '}
              <code style={{ 
                backgroundColor: '#fff',
                padding: '1px 5px',
                borderRadius: '2px',
                border: '1px solid #dee2e6',
                fontSize: '11px'
              }}>
                [{actionResult.buttons.join(', ')}]
              </code>
            </span>
          )}
          
          {actionResult.direction && (
            <span>
              <strong style={{ color: '#444' }}>Direction:</strong>{' '}
              <code style={{ 
                backgroundColor: '#fff',
                padding: '1px 5px',
                borderRadius: '2px',
                border: '1px solid #dee2e6',
                fontSize: '11px'
              }}>
                {actionResult.direction}
              </code>
            </span>
          )}
          
          {actionResult.location && (
            <span>
              <strong style={{ color: '#444' }}>Location:</strong>{' '}
              <code style={{ 
                backgroundColor: '#fff',
                padding: '1px 5px',
                borderRadius: '2px',
                border: '1px solid #dee2e6',
                fontSize: '11px',
                maxWidth: '200px',
                display: 'inline-block',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                verticalAlign: 'middle'
              }}>
                {actionResult.location}
              </code>
            </span>
          )}
          
          {/* Timing Info */}
          {actionResult.coord_time !== null && actionResult.coord_time !== undefined && (
            <span>
              <strong style={{ color: '#444' }}>Coord:</strong> {actionResult.coord_time.toFixed(3)}s
            </span>
          )}
          {actionResult.exec_time !== null && actionResult.exec_time !== undefined && (
            <span>
              <strong style={{ color: '#444' }}>Exec:</strong> {actionResult.exec_time.toFixed(3)}s
            </span>
          )}
          
          {/* Error if any */}
          {actionResult.error && (
            <span style={{ color: '#dc3545', flex: '1 1 100%' }}>
              <strong>Error:</strong> {actionResult.error}
            </span>
          )}
        </div>
      )}
    </div>
  );
};

export default ActionDetails;

