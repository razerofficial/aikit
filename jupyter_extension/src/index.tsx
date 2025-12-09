import { JupyterFrontEnd, JupyterFrontEndPlugin } from '@jupyterlab/application';
import { ILayoutRestorer, ILabShell } from '@jupyterlab/application';
import { IDocumentManager } from '@jupyterlab/docmanager';
import { MainAreaWidget, ReactWidget } from '@jupyterlab/apputils';
import * as React from 'react';
import { Home } from './Home';
import '../style/index.css';


const PLUGIN_ID = 'jlab-home:plugin';

const plugin: JupyterFrontEndPlugin<void> = {
  id: PLUGIN_ID,
  autoStart: true,
  requires: [IDocumentManager, ILabShell],
  optional: [ILayoutRestorer],
  activate: (app: JupyterFrontEnd, docManager: IDocumentManager, labShell: ILabShell, restorer: ILayoutRestorer | null) => {
    const open = async (path: string) => {
      await app.commands.execute('docmanager:open', { path });
    };

    const content = ReactWidget.create(<Home open={open} />);
    const widget = new MainAreaWidget({ content });
    widget.id = 'home';
    widget.title.label = 'Home';
    widget.title.closable = true;

    // Make it appear on first run, and be restorable thereafter
    restorer?.add(widget, widget.id);
    labShell.add(widget, 'main', { activate: true });
  }
};

export default plugin;

