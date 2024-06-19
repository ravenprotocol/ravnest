# Ravnest Documentation

Instructions on how you can set up Ravnest's documentation locally on your system.

## Setup

Switch to the ```docs``` folder.

```bash
cd docs/
```

### Installing Dependencies

This will set up all documentation specific Sphinx related modules:

```bash
pip install -r requirements.txt
```

## Rendering

Inside the ```docs``` folder, simply run the following command to generate the html, css and javascript files for rendering the webpage:

```bash
make html
```

Now open ```docs/_build/html/index.html``` file using Live Server on VS Code to view the full documentation in your browser.

> **_NOTE:_**  If you face any issues with the webpage render not updating as you add new content or modify to the scripts, it would be a good idea to delete ```_build``` folder, run ```make clean``` and try ```make html``` again.

## Contributing to Documentation

1. Set up the docs on your system locally.
2. Make your proposed changes.
3. Run ```make_html``` command inside ```docs``` folder and check the resultant logs. Please make sure there are no errors or warnings.
4. Create Pull Request on GitHub. 
5. We will review and merge.





