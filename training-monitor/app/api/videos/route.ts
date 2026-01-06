/**
 * API route to serve video files from the logs directory.
 */

import { NextRequest, NextResponse } from 'next/server';
import { existsSync, createReadStream, statSync } from 'fs';
import { join } from 'path';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const path = searchParams.get('path');

    if (!path) {
      return NextResponse.json(
        { error: 'Video path is required' },
        { status: 400 }
      );
    }

    // Security: Prevent directory traversal attacks
    if (path.includes('..') || path.startsWith('/')) {
      return NextResponse.json(
        { error: 'Invalid video path' },
        { status: 400 }
      );
    }

    // Construct full path
    // process.cwd() returns training-monitor directory
    // Go up one level to tinker-cookbook root, then access the logs directory
    const projectRoot = join(process.cwd(), '..');
    const videoPath = join(projectRoot, path);

    console.log('Video request:', { path, projectRoot, videoPath });

    // Check if file exists
    if (!existsSync(videoPath)) {
      return NextResponse.json(
        { error: 'Video file not found' },
        { status: 404 }
      );
    }

    // Get file stats
    const stats = statSync(videoPath);
    const fileSize = stats.size;

    // Handle range requests for video streaming
    const range = request.headers.get('range');
    
    if (range) {
      // Parse range header
      const parts = range.replace(/bytes=/, '').split('-');
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
      const chunkSize = (end - start) + 1;

      // Create read stream for the requested range
      const stream = createReadStream(videoPath, { start, end });
      
      // Convert Node.js stream to Web stream
      const readableStream = new ReadableStream({
        start(controller) {
          stream.on('data', (chunk) => {
            controller.enqueue(new Uint8Array(chunk));
          });
          stream.on('end', () => {
            controller.close();
          });
          stream.on('error', (err) => {
            controller.error(err);
          });
        },
      });

      return new NextResponse(readableStream, {
        status: 206,
        headers: {
          'Content-Range': `bytes ${start}-${end}/${fileSize}`,
          'Accept-Ranges': 'bytes',
          'Content-Length': chunkSize.toString(),
          'Content-Type': 'video/mp4',
        },
      });
    } else {
      // No range request - send entire file
      const stream = createReadStream(videoPath);
      
      // Convert Node.js stream to Web stream
      const readableStream = new ReadableStream({
        start(controller) {
          stream.on('data', (chunk) => {
            controller.enqueue(new Uint8Array(chunk));
          });
          stream.on('end', () => {
            controller.close();
          });
          stream.on('error', (err) => {
            controller.error(err);
          });
        },
      });

      return new NextResponse(readableStream, {
        status: 200,
        headers: {
          'Content-Length': fileSize.toString(),
          'Content-Type': 'video/mp4',
          'Accept-Ranges': 'bytes',
        },
      });
    }
  } catch (error: any) {
    console.error('Error serving video:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to serve video' },
      { status: 500 }
    );
  }
}

