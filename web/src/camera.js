// The webcam and MediaPipe's hand landmarker. Video frames never leave the browser: MediaPipe
// runs locally in WebAssembly, and only the 21 landmark positions reach the game.

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.0/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const LINE_COLOUR = 'rgba(255, 255, 255, 0.85)';
const POINT_COLOUR = '#eb6834';

/** @typedef {import('./gesture.js').Point} Point */
/** @typedef {(points: Point[] | null, aspect: number) => void} FrameHandler */

export class HandCamera {
  /**
   * @param {HTMLVideoElement} video
   * @param {HTMLCanvasElement} canvas
   * @param {import('@mediapipe/tasks-vision').HandLandmarker} landmarker
   * @param {MediaStream} stream
   * @param {{start: number, end: number}[]} connections
   */
  constructor(video, canvas, landmarker, stream, connections) {
    this.video = video;
    this.canvas = canvas;
    this.landmarker = landmarker;
    this.stream = stream;
    this.connections = connections;
    this.running = false;
    this.lastTime = -1;
  }

  /**
   * Ask for the webcam and load the hand landmarker.
   * @param {HTMLVideoElement} video
   * @param {HTMLCanvasElement} canvas
   * @returns {Promise<HandCamera>}
   */
  static async open(video, canvas) {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    try {
      video.srcObject = stream;
      await video.play();
      const { FilesetResolver, HandLandmarker } = await import('@mediapipe/tasks-vision');
      const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
      /** @param {'GPU' | 'CPU'} delegate */
      const create = (delegate) =>
        HandLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate },
          runningMode: 'VIDEO',
          numHands: 1,
        });
      // Some browsers expose WebGL but fail to run MediaPipe's GPU path
      const landmarker = await create('GPU').catch(() => create('CPU'));
      return new HandCamera(video, canvas, landmarker, stream, HandLandmarker.HAND_CONNECTIONS);
    } catch (error) {
      for (const track of stream.getTracks()) {
        track.stop();
      }
      throw error;
    }
  }

  /**
   * Read every new video frame and pass the landmarks of the hand, if any, to the handler.
   * @param {FrameHandler} onFrame
   */
  start(onFrame) {
    this.running = true;
    const step = () => {
      if (!this.running) {
        return;
      }
      const { video } = this;
      if (video.readyState >= 2 && video.currentTime !== this.lastTime) {
        this.lastTime = video.currentTime;
        const result = this.landmarker.detectForVideo(video, performance.now());
        const hand = result.landmarks[0];
        /** @type {Point[] | null} */
        const points = hand ? hand.map((landmark) => [landmark.x, landmark.y]) : null;
        this.draw(points);
        onFrame(points, video.videoHeight / video.videoWidth);
      }
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  /**
   * Draw the hand skeleton over the video.
   * @param {Point[] | null} points
   */
  draw(points) {
    const { canvas, video } = this;
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
    const context = canvas.getContext('2d');
    if (context === null) {
      return;
    }
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (points === null) {
      return;
    }
    const scale = canvas.width / 320;
    context.lineWidth = 2 * scale;
    context.strokeStyle = LINE_COLOUR;
    context.beginPath();
    for (const { start, end } of this.connections) {
      context.moveTo(points[start][0] * canvas.width, points[start][1] * canvas.height);
      context.lineTo(points[end][0] * canvas.width, points[end][1] * canvas.height);
    }
    context.stroke();
    context.fillStyle = POINT_COLOUR;
    for (const [x, y] of points) {
      context.beginPath();
      context.arc(x * canvas.width, y * canvas.height, 2.5 * scale, 0, 2 * Math.PI);
      context.fill();
    }
  }

  stop() {
    this.running = false;
    for (const track of this.stream.getTracks()) {
      track.stop();
    }
    this.landmarker.close();
  }
}
