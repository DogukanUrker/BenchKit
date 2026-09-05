# mc-arena viewer

The renderer behind BenchKit's `mc-arena` suite is
[prismarine-viewer](https://github.com/mc-bench/prismarine-viewer) (MIT),
pinned to the commit in `version.json` and bundled for the browser here. The
built bundle in `../dist` is committed, so running BenchKit needs neither Node
nor a network: only Python, Chromium and Docker.

## What is in `../dist`

| File | Where it comes from |
| --- | --- |
| `viewer.js` | `entry.js`, bundled with prismarine-viewer and three.js |
| `worker.js` | prismarine-viewer's chunk-meshing web worker |
| `atlas.png` | the block texture atlas prismarine-viewer prerenders from [mc-assets](https://github.com/mc-bench/mc-assets) (MIT) |
| `blockStates.json.gz` | the block-state models for the pinned version, gzipped |
| `viewer.html` | the page BenchKit opens; it holds nothing model-written |
| `version.json` | the pinned Minecraft version and viewer commit |

`entry.js` also owns the camera rig: the build volume, the three camera
positions, the lens and the image size. Changing any of them makes new
screenshots incomparable with old ones, so treat it the way the prompt set is
treated and say so in the run notes.

The block id table `mc-arena` validates against
(`../../datasets/mc_blocks_1_20_1.jsonl`) is extracted from
[mc-data-files](https://github.com/mc-bench/mc-data-files) (MIT) for the same
version, so a build cannot be called valid with an id the renderer would then
silently drop.

## Rebuilding

```bash
npm install --ignore-scripts   # --ignore-scripts: the viewer's own prepare
npm run build                  # step rebuilds atlases and needs cairo
```

The output lands in `../dist/*.js`. `atlas.png` and `blockStates.json.gz` are
produced by prismarine-viewer's own `viewer/prerender.js` (which needs the
`canvas` native module and therefore cairo); regenerate them only when moving
to a different Minecraft version:

```bash
git clone https://github.com/mc-bench/prismarine-viewer
cd prismarine-viewer && npm install          # runs prerender + webpack
cp public/textures/<version>.png     ../dist/atlas.png
gzip -9 -c public/blocksStates/<version>.json > ../dist/blockStates.json.gz
```

`minecraft-data` ships every Minecraft version ever released. `webpack.config.js`
drops everything the pinned version does not map to - without that filter the
worker bundle is 116 MB rather than 3 MB.
