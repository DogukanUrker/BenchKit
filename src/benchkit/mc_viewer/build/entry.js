/* eslint-env browser */
// BenchKit's mc-arena viewer: prismarine-viewer, pinned to one Minecraft
// version and driven from a fixed camera rig so every model's build is
// photographed the same way.
//
// The page is BenchKit's own code, not the model's - a model only supplies a
// list of block positions. Rendering happens in the browser because that is
// where prismarine-viewer's meshing workers and WebGL renderer live.
global.THREE = require('three')

const { Viewer } = require('prismarine-viewer/viewer')
const { Vec3 } = require('vec3')

const VERSION = require('./version.json').minecraft_version
const mcData = require('minecraft-data')(VERSION)
const Chunk = require('prismarine-chunk')(VERSION)

const AIR = new Set(['air', 'cave_air', 'void_air'])

// The build volume, the camera rig and the image size are frozen: two models
// are only comparable if their screenshots were taken from the same place.
const SIZE = 32
const CENTER = new Vec3(SIZE / 2, SIZE / 2, SIZE / 2)
// A long lens from far away: the whole build volume fits the frame with very
// little perspective, which is what makes two builds comparable side by side.
const DISTANCE = 112
const FOV = 25
const IMAGE = 640

// Unit directions from the build's centre towards the camera, and the world
// axis that points up in the resulting image. Straight down needs an explicit
// up vector: with the default one the view direction and up are parallel and
// the framing degenerates.
const VIEWS = [
  { name: 'iso', label: 'Isometric', dir: [0.7, 0.55, 0.7], up: [0, 1, 0] },
  { name: 'side', label: 'Side', dir: [0, 0.18, 1], up: [0, 1, 0] },
  { name: 'top', label: 'Top-down', dir: [0, 1, 0], up: [0, 0, -1] }
]

const normalize = ([x, y, z]) => {
  const length = Math.hypot(x, y, z) || 1
  return [x / length, y / length, z / length]
}

/** Resolve a `minecraft:xxx` id to a default block state, or null. */
function stateId (name) {
  const plain = String(name).replace(/^minecraft:/, '').toLowerCase()
  if (AIR.has(plain)) return null
  const block = mcData.blocksByName[plain]
  if (!block) return null
  const state = block.defaultState ?? block.minStateId
  return typeof state === 'number' ? state : null
}

/** Fill the 2x2 chunk columns the 32-block volume spans. */
function buildChunks (blocks) {
  const columns = new Map()
  const column = (cx, cz) => {
    const key = `${cx},${cz}`
    if (!columns.has(key)) {
      columns.set(key, new Chunk({ minY: mcData.version.minY ?? -64, worldHeight: 384 }))
    }
    return columns.get(key)
  }
  // Every column the volume touches exists even when empty, so the mesher is
  // asked about the whole box and an empty build still renders as an empty box.
  for (let cx = 0; cx < SIZE; cx += 16) {
    for (let cz = 0; cz < SIZE; cz += 16) column(cx, cz)
  }

  let placed = 0
  let unknown = 0
  for (const block of blocks) {
    const x = Math.trunc(block.x)
    const y = Math.trunc(block.y)
    const z = Math.trunc(block.z)
    if (!(x >= 0 && x < SIZE && y >= 0 && y < SIZE && z >= 0 && z < SIZE)) continue
    const id = stateId(block.block)
    if (id === null) {
      unknown += 1
      continue
    }
    const cx = Math.floor(x / 16) * 16
    const cz = Math.floor(z / 16) * 16
    column(cx, cz).setBlockStateId(new Vec3(x & 15, y, z & 15), id)
    placed += 1
  }
  return { columns, placed, unknown }
}

async function loadAssets (viewer) {
  const [atlas, states] = await Promise.all([
    fetch('atlas.png').then(response => response.blob()).then(blob => new Promise(resolve => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result)
      reader.readAsDataURL(blob)
    })),
    // Shipped gzipped: the raw block-state models are ~10 MB of JSON.
    fetch('blockStates.json.gz')
      .then(response => response.body.pipeThrough(new DecompressionStream('gzip')))
      .then(stream => new Response(stream).json())
  ])
  viewer.world.texturesDataUrl = atlas
  viewer.world.blockStatesData = states
}

function place (camera, view) {
  const [x, y, z] = normalize(view.dir)
  camera.position.set(
    CENTER.x + x * DISTANCE,
    CENTER.y + y * DISTANCE,
    CENTER.z + z * DISTANCE
  )
  camera.up.set(...view.up)
  camera.lookAt(CENTER.x, CENTER.y, CENTER.z)
  camera.updateProjectionMatrix()
}

async function main () {
  const parameters = new URLSearchParams(window.location.search)
  const blocks = await fetch(parameters.get('build') || 'build.json').then(r => r.json())

  const canvas = document.createElement('canvas')
  const renderer = new global.THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true })
  renderer.setPixelRatio(1)
  renderer.setSize(IMAGE, IMAGE)

  const viewer = new Viewer(renderer)
  viewer.scene.background = new global.THREE.Color('#dfe6ee')
  viewer.camera = new global.THREE.PerspectiveCamera(FOV, 1, 0.1, 1000)
  // The atlas and block-state models ship with BenchKit, so they are handed
  // to the renderer before setVersion goes looking for them on a web server.
  await loadAssets(viewer)
  if (!viewer.setVersion(VERSION)) throw new Error(`prismarine-viewer does not support ${VERSION}`)

  const { columns, placed, unknown } = buildChunks(Array.isArray(blocks) ? blocks : [])
  for (const [key, chunk] of columns) {
    const [cx, cz] = key.split(',').map(Number)
    viewer.addColumn(cx, cz, chunk.toJson())
  }
  await viewer.world.waitForChunksToRender()

  const shots = document.getElementById('views')
  for (const view of VIEWS) {
    place(viewer.camera, view)
    viewer.update()
    renderer.render(viewer.scene, viewer.camera)
    const figure = document.createElement('figure')
    const image = document.createElement('img')
    image.id = `view-${view.name}`
    image.width = IMAGE
    image.height = IMAGE
    image.src = canvas.toDataURL('image/png')
    const caption = document.createElement('figcaption')
    caption.textContent = view.label
    figure.append(image, caption)
    shots.append(figure)
  }
  await Promise.all([...document.images].map(image => image.decode().catch(() => {})))

  window.__mcarena = {
    ok: true,
    version: VERSION,
    placed,
    unresolved: unknown,
    views: VIEWS.map(view => view.name),
    image_size: IMAGE
  }
}

main().catch(error => {
  window.__mcarena = { ok: false, error: String((error && error.stack) || error) }
})
