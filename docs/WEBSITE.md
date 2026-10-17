# Project website

[index.html](index.html) is the bilingual interactive project page supplied with this release. Its images, demonstration data, styles and scripts are embedded in the file. The English and Chinese code buttons link to the complete GitHub repository.

## Preview locally

From the repository root, run:

```bash
python -m http.server 8000 --directory docs
```

Open `http://localhost:8000/` or `http://localhost:8000/?lang=zh-CN` in a browser. Check the leaderboard, language switch, selection demonstration and data downloads.

## Publish with GitHub Pages

This public repository serves the project page from `main` and `/docs`. To configure or restore the deployment:

1. Open the repository's Settings > Pages.
2. Set the publishing source to **Deploy from a branch**.
3. Select **main** and **/docs**, then save.
4. Wait for the Pages deployment to complete and use the URL reported by GitHub.

The project URL is `https://lingbei-meng.github.io/SLDBench-v2/`. Deployment status is available in the repository Actions tab.

The [.nojekyll](.nojekyll) marker serves the supplied static page without a Jekyll build. No model API key or backend is required by this page.

See the [GitHub Pages publishing-source documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site) for current availability and configuration details.
