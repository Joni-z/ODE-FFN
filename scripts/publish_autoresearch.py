"""Publish verified local ODE-FFN commits directly, without remote-machine relays."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess


ROOT=Path(__file__).resolve().parents[1]
ACCEPTED_NAMES={'Zhizhe Zhang','Jonathan Zhang'}
REMOTES={'https://github.com/Joni-z/ODE-FFN.git','git@github.com:Joni-z/ODE-FFN.git'}


def git(*arguments: str) -> str:
    result=subprocess.run(['git','-C',str(ROOT),*arguments],text=True,capture_output=True,timeout=60)
    if result.returncode:
        raise RuntimeError(f'git {arguments[0]} failed: {result.stderr.strip()}')
    return result.stdout.strip()


def main() -> None:
    if git('remote','get-url','origin') not in REMOTES:
        raise RuntimeError('origin must be the canonical Joni-z/ODE-FFN repository')
    if git('status','--porcelain'):
        raise RuntimeError('commit reviewed changes before publication; preserve the worktree')
    head=git('rev-parse','HEAD')
    print(json.dumps(dict(phase='fetch',sha=head)),flush=True)
    git('fetch','origin','main')
    base=git('rev-parse','origin/main')
    if git('merge-base',base,head)!=base:
        raise RuntimeError('remote main advanced or histories diverged; reconcile before a normal push')
    for record in git('log','--format=%an%x09%cn',base+'..'+head).splitlines():
        author,committer=record.split('\t')
        if author not in ACCEPTED_NAMES or committer not in ACCEPTED_NAMES:
            raise RuntimeError('unpublished commits must use Zhizhe Zhang or Jonathan Zhang')
    if base!=head:
        print(json.dumps(dict(phase='push',sha=head)),flush=True)
        git('push','origin','HEAD:main')
    actual=git('ls-remote','origin','refs/heads/main').split()[0]
    if actual!=head:
        raise RuntimeError('remote main differs from the expected published SHA; inspect before retrying')
    print(json.dumps(dict(status='PUBLISHED_AND_VERIFIED',sha=head,transport='local Git')),flush=True)


if __name__=='__main__':
    main()
