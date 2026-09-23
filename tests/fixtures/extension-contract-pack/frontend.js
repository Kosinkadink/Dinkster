export const frontendExtension = {
  activate(context) {
    const statusId = 'dinkster-extension-contract-fixture.status';
    let routeMessage = 'Loading third-party pack route';
    let latest;
    const refresh = () => context.invalidateHostUi(statusId);

    context.hostUi(statusId, 'status.trailing', () => ({
      version: 1,
      root: {
        kind: 'group', key: 'extension-contract', direction: 'column', children: [
          { kind: 'status', key: 'route', tone: routeMessage.endsWith('ready') ? 'success' : 'neutral', text: routeMessage },
          { kind: 'status', key: 'event', live: 'polite', tone: latest ? 'success' : 'neutral',
            text: latest
              ? `Custom fixture execution: ${latest.data.width} x ${latest.data.height} (mean ${latest.data.mean.toFixed(3)})`
              : 'Waiting for custom fixture execution' },
        ],
      },
    }), 0, 'Extension contract proof');

    context.eventConsumer('dinkster-extension-contract-fixture.event', (event) => {
      latest = event;
      refresh();
    });
    context.canvasLayer('dinkster-extension-contract-fixture.canvas', {
      id: 'dinkster-extension-contract-fixture.canvas',
      position: 'foreground',
      draw({ context: canvas, nodes }) {
        const proof = nodes.find((node) => node.id === 'proof');
        globalThis.__dinksterExtensionContractCanvas = {
          draws: (globalThis.__dinksterExtensionContractCanvas?.draws ?? 0) + 1,
          nodes: nodes.map((node) => node.id),
        };
        if (!proof) return;
        canvas.strokeStyle = '#ff4fd8';
        canvas.lineWidth = 3;
        canvas.setLineDash([8, 5]);
        canvas.strokeRect(proof.x - 8, proof.y - 8, proof.width + 16, proof.height + 16);
      },
    });
    context.nodeDecoration('dinkster-extension-contract-fixture.node-decoration', {
      id: 'dinkster-extension-contract-fixture.node-decoration',
      decorate(node) {
        if (node.id !== 'proof') return;
        return {
          badges: [{
            id: 'dinkster-extension-contract-fixture.proof-badge',
            glyph: 'Pack',
            variant: 'label',
            interactive: false,
            color: '#7b3fb2',
          }],
        };
      },
    });
    context.onDispose(() => { latest = undefined; });
    void context.queryRoute('extension-contract').then((value) => {
      routeMessage = value.message;
      refresh();
    }).catch(() => {
      if (context.signal.aborted) return;
      routeMessage = 'Third-party pack route unavailable';
      refresh();
    });
  },
};
