"""Validate bilingual navigation and executable snippets without running installs."""
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class DocumentationTest(unittest.TestCase):
    def test_language_pairs_and_local_links(self):
        documents = [*ROOT.glob('*.md'), *(ROOT / 'docs').rglob('*.md')]
        for path in documents:
            if path.name.endswith('.en.md'):
                counterpart = path.with_name(path.name.replace('.en.md', '.md'))
            else:
                counterpart = path.with_suffix('.en.md')
            self.assertTrue(counterpart.is_file(), path)
            content = path.read_text()
            targets = re.findall(r'\]\(([^)]+)\)', content)
            targets += re.findall(r'<img[^>]+src="([^"]+)"', content)
            for target in targets:
                if target.startswith(('https://', 'http://', '#')):
                    continue
                self.assertTrue((path.parent / target.split('#')[0]).exists(), (path, target))

    def test_translations_have_identical_commands(self):
        for path in (ROOT / 'docs/deployment').glob('*.md'):
            if path.name.endswith('.en.md'):
                continue
            translated = path.with_suffix('.en.md')
            commands = re.findall(r'```bash\n(.*?)\n```', path.read_text(), re.S)
            self.assertEqual(commands, re.findall(r'```bash\n(.*?)\n```', translated.read_text(), re.S), path)
            for command in commands:
                subprocess.run(['bash', '-n'], input=command, text=True, check=True)

if __name__ == '__main__':
    unittest.main()
