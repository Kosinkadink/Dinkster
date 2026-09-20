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
              ? `Custom dev.image execution: ${latest.data.width} x ${latest.data.height} (mean ${latest.data.mean.toFixed(3)})`
              : 'Waiting for custom dev.image execution' },
        ],
      },
    }), 0, 'Extension contract proof');

    context.eventConsumer('dinkster-extension-contract-fixture.event', (event) => {
      latest = event;
      refresh();
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
