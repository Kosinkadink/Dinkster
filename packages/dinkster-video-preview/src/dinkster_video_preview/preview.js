export const frontendExtension = {
  activate(context) {
    const statusId = 'dinkster-video-preview.preview.status';
    let policy;
    let latest;
    let policyState = 'Loading preview policy';
    const refresh = () => context.invalidateHostUi(statusId);

    context.hostUi(statusId, 'status.trailing', () => ({
      version: 1,
      root: {
        kind: 'group', key: 'video-preview', direction: 'column', children: [
          { kind: 'text', key: 'identity',
            text: `Video preview: ${context.identity.connection || 'local'} [${context.identity.snapshotDigest.slice(7, 19)}]` },
          { kind: 'text', key: 'policy',
            text: policy
              ? `Preview policy: ${policy.defaultFps} fps | up to ${policy.maxFrames} frames, ${policy.maxWidth} px`
              : policyState },
          { kind: 'status', key: 'metadata', live: 'polite',
            tone: latest ? 'success' : 'neutral',
            text: latest
              ? `${latest.runtimeNodeId || 'Video'}: ${latest.data.width} x ${latest.data.height} | ${latest.data.fps} fps | ${latest.data.frameCount} frames`
              : 'Waiting for video preview initialization' },
          ...(latest && policy && (latest.data.frameCount > policy.maxFrames || latest.data.width > policy.maxWidth)
            ? [{ kind: 'status', key: 'limit', tone: 'warning', text: 'Video exceeds preview policy limits' }]
            : []),
        ],
      },
    }), 0, 'Video preview');

    context.eventConsumer('dinkster-video-preview.preview.initialized', (event) => {
      latest = event;
      refresh();
    });
    context.onDispose(() => { latest = undefined; policy = undefined; });
    void context.queryRoute('preview-policy').then((value) => {
      policy = value;
      refresh();
    }).catch(() => {
      if (context.signal.aborted) return;
      policyState = 'Preview policy unavailable; native video remains available';
      refresh();
    });
  },
};
