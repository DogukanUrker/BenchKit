// Two bundles: the page itself and prismarine-viewer's meshing worker. Both
// are pinned to one Minecraft version - minecraft-data ships every version
// ever released, and bundling all of them turns a 4 MB worker into 116 MB.
const path = require('path')
const webpack = require('webpack')

const VERSION = require('./version.json').minecraft_version
const DIST = path.resolve(__dirname, '../dist')

// minecraft-data ships every Minecraft version ever released. Only the files
// dataPaths maps for the pinned version are kept - without this the worker
// bundle is 116 MB instead of 1 MB.
const dataPaths = require('minecraft-data/minecraft-data/data/dataPaths.json')

const pinnedPaths = new Set(
  Object.entries(dataPaths.pc[VERSION]).map(([name, dir]) => `${dir}/${name}.json`)
)
// minecraft-data's own index loads both editions' common tables at import
// time and indexes them eagerly, so these stay in whatever the pin is.
const COMMON = ['protocolVersions', 'versions', 'legacy', 'features']
for (const edition of ['pc', 'bedrock']) {
  for (const name of COMMON) pinnedPaths.add(`${edition}/common/${name}.json`)
}
// prismarine-viewer's mesher reads its tint tables from a hard-coded version
// of its own, whatever world version it is rendering.
for (const [version, name] of [['1.16.2', 'tints']]) {
  pinnedPaths.add(`${dataPaths.pc[version][name]}/${name}.json`)
}

/** Drop every minecraft-data JSON the pinned version does not map to. */
const pinData = (request, callback) => {
  if (request.context.includes('minecraft-data') && request.request.endsWith('.json')) {
    const relative = request.request.replace(/^.*?((pc|bedrock)\/)/, '$1')
    // Schemas are tiny and are required by name, not by version.
    if (!request.request.includes('/schemas/') && !pinnedPaths.has(relative)) {
      callback(null, [])
      return
    }
  }
  callback()
}

const shared = {
  mode: process.env.BENCHKIT_MC_VIEWER_DEV ? 'development' : 'production',
  devtool: false,
  resolve: { fallback: { assert: require.resolve('assert/'), zlib: false, fs: false, path: false } },
  plugins: [
    new webpack.ProvidePlugin({ process: 'process/browser' }),
    new webpack.ProvidePlugin({ Buffer: ['buffer', 'Buffer'] })
  ],
  performance: { hints: false }
}

module.exports = [
  {
    ...shared,
    entry: path.resolve(__dirname, './entry.js'),
    output: { path: DIST, filename: 'viewer.js' },
    plugins: [
      ...shared.plugins,
      // The browser build of prismarine-viewer's asset loaders; the default
      // reads from disk with node's fs.
      new webpack.NormalModuleReplacementPlugin(
        /viewer[/\\]lib[/\\]utils/,
        './utils.web.js'
      )
    ],
    externals: [pinData]
  },
  {
    ...shared,
    entry: path.resolve(__dirname, './node_modules/prismarine-viewer/viewer/lib/worker.js'),
    output: { path: DIST, filename: 'worker.js' },
    externals: [pinData]
  }
]
