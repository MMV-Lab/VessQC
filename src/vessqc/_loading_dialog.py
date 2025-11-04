"""
Loading dialog for VessQC

This module provides a dialog for selecting and loading datasets
with priority-based ordering.

Imports
-------
qtpy.QtWidgets, qtpy.QtCore

Exports
-------
LoadingDialog
"""

# Copyright © Peter Lampen, Lennart Kowitz, ISAS Dortmund, 2025

from pathlib import Path
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QMessageBox,
    QProgressBar,
)
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ._data_manager import DataManager, DatasetTriplet


class LoadingDialog(QDialog):
    """
    Dialog for selecting and loading datasets
    
    Signals
    -------
    dataset_selected : DatasetTriplet
        Emitted when a dataset is selected for loading
    """
    
    dataset_selected = Signal(object)  # DatasetTriplet
    
    def __init__(self, data_manager: 'DataManager', parent=None):
        """
        Initialize the loading dialog
        
        Parameters
        ----------
        data_manager : DataManager
            Data manager instance
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self.data_manager = data_manager
        self.selected_triplet: Optional['DatasetTriplet'] = None
        
        self.setWindowTitle('Load Dataset')
        self.setMinimumWidth(800)
        self.setMinimumHeight(500)
        
        self._setup_ui()
        
        # Auto-refresh on open if directory is set
        if self.data_manager.data_directory:
            self._refresh_datasets()
            # Queue unsegmented datasets for background processing
            if parent and hasattr(parent, '_queue_unsegmented_datasets'):
                parent._queue_unsegmented_datasets()
        else:
            self._update_dataset_list()
    
    def _setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Directory selection
        dir_layout = QHBoxLayout()
        self.dir_label = QLabel('Data Directory: Not set')
        dir_layout.addWidget(self.dir_label)
        
        btn_select_dir = QPushButton('Select Directory')
        btn_select_dir.clicked.connect(self._select_directory)
        dir_layout.addWidget(btn_select_dir)
        
        btn_refresh = QPushButton('Refresh')
        btn_refresh.clicked.connect(self._refresh_datasets)
        dir_layout.addWidget(btn_refresh)
        
        layout.addLayout(dir_layout)
        
        # Info label
        self.info_label = QLabel('Datasets ordered by highest uncertainty (excluding Noise), then size')
        self.info_label.setStyleSheet('color: gray; font-style: italic;')
        layout.addWidget(self.info_label)
        
        # Queue status label
        self.queue_label = QLabel('')
        self.queue_label.setStyleSheet('color: blue; font-weight: bold;')
        layout.addWidget(self.queue_label)
        
        # Progress bar for priority calculation
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat('Calculating priorities...')
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Dataset list
        self.dataset_list = QListWidget()
        self.dataset_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.dataset_list)
        
        # Status label
        self.status_label = QLabel('')
        self.status_label.setStyleSheet('color: gray;')
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        btn_load = QPushButton('Load Selected')
        btn_load.clicked.connect(self._load_selected)
        button_layout.addWidget(btn_load)
        
        btn_cancel = QPushButton('Cancel')
        btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(btn_cancel)
        
        layout.addLayout(button_layout)
        
        # Update directory label if directory is set
        if self.data_manager.data_directory:
            self.dir_label.setText(f'Data Directory: {self.data_manager.data_directory}')
    
    def _select_directory(self):
        """Open dialog to select data directory"""
        directory = QFileDialog.getExistingDirectory(
            self,
            'Select Data Directory',
            str(self.data_manager.data_directory) if self.data_manager.data_directory else ''
        )
        
        if directory:
            # Check if directory actually changed
            old_directory = self.data_manager.data_directory
            new_directory = Path(directory)
            directory_changed = (old_directory != new_directory)
            
            if self.data_manager.set_data_directory(new_directory):
                self.dir_label.setText(f'Data Directory: {directory}')
                
                # Clear segmentation queue if directory changed
                if directory_changed and self.parent() and hasattr(self.parent(), 'segmentation_worker'):
                    print(f"DEBUG: Directory changed from {old_directory} to {new_directory}")
                    print(f"DEBUG: Clearing segmentation queue...")
                    self.parent().segmentation_worker.clear_queue()
                
                self._refresh_datasets()
                # Queue unsegmented datasets for background processing
                if self.parent() and hasattr(self.parent(), '_queue_unsegmented_datasets'):
                    self.parent()._queue_unsegmented_datasets()
            else:
                QMessageBox.warning(self, 'Error', 'Could not set data directory')
    
    def _refresh_datasets(self):
        """Refresh the dataset list"""
        print(f"\nDEBUG: LoadingDialog._refresh_datasets() called")
        
        if not self.data_manager.data_directory:
            QMessageBox.information(self, 'No Directory', 
                                   'Please select a data directory first')
            return
        
        # Check for changes
        print(f"DEBUG: Syncing with directory...")
        added, removed = self.data_manager.sync_with_directory()
        
        if added or removed:
            msg = []
            if added:
                msg.append(f"New datasets: {', '.join(added)}")
            if removed:
                msg.append(f"Removed datasets: {', '.join(removed)}")
            self.status_label.setText(' | '.join(msg))
            print(f"DEBUG: Directory changes - added: {len(added)}, removed: {len(removed)}")
        else:
            self.status_label.setText('No changes detected')
            print(f"DEBUG: No directory changes detected")
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.dataset_list.setEnabled(False)
        
        # Queue unsegmented datasets for background processing
        if self.parent() and hasattr(self.parent(), '_queue_unsegmented_datasets'):
            print(f"DEBUG: Queueing unsegmented datasets...")
            self.parent()._queue_unsegmented_datasets()
        
        # Calculate priorities in background
        print(f"DEBUG: Starting priority calculation...")
        self.data_manager.calculate_priorities(
            callback=self._on_priorities_calculated,
            threaded=True
        )
    
    def _on_priorities_calculated(self):
        """Called when priority calculation is complete"""
        print(f"\nDEBUG: LoadingDialog._on_priorities_calculated() called")
        print(f"DEBUG:   Total segments in all_segment_priorities: {len(self.data_manager.all_segment_priorities)}")
        print(f"DEBUG:   Total datasets: {len(self.data_manager.datasets)}")
        
        # This is called from the worker thread, so we need to be careful
        # For now, just update the list (Qt should handle cross-thread signals)
        self._update_dataset_list()
        self.progress_bar.setVisible(False)
        self.dataset_list.setEnabled(True)
        
        # Update status
        datasets_with_seg = sum(1 for d in self.data_manager.datasets if d.has_segmentation)
        datasets_without_seg = len(self.data_manager.datasets) - datasets_with_seg
        
        print(f"DEBUG:   Datasets with segmentation: {datasets_with_seg}")
        print(f"DEBUG:   Datasets without segmentation: {datasets_without_seg}")
        
        status_text = f'Found {len(self.data_manager.datasets)} datasets'
        if datasets_without_seg > 0:
            status_text += f' ({datasets_without_seg} calculating...)'
        self.status_label.setText(status_text)
        
        # Update queue status
        if self.parent() and hasattr(self.parent(), 'segmentation_worker'):
            worker = self.parent().segmentation_worker
            queue_size = worker.get_queue_size()
            current = worker.current_dataset
            queued_count = len(worker._queued_datasets)
            
            print(f"DEBUG:   Queue status check:")
            print(f"DEBUG:     current_dataset: {current}")
            print(f"DEBUG:     queue.qsize(): {queue_size}")
            print(f"DEBUG:     _queued_datasets: {queued_count} items: {worker._queued_datasets}")
            
            # Check if there are datasets that still need segmentation
            # (either in queue or not yet calculated)
            if current or queue_size > 0 or datasets_without_seg > 0:
                self.queue_label.setText(f'🔄 Calculating segmentations in background...')
            else:
                self.queue_label.setText('✅ All segmentations complete')
        else:
            self.queue_label.setText('')
    
    def _update_dataset_list(self):
        """Update the dataset list widget to show datasets"""
        print(f"\nDEBUG: LoadingDialog._update_dataset_list() called")
        print(f"DEBUG:   Total datasets: {len(self.data_manager.datasets)}")
        
        self.dataset_list.clear()
        
        if not self.data_manager.datasets:
            item = QListWidgetItem('No datasets found')
            item.setFlags(Qt.NoItemFlags)
            self.dataset_list.addItem(item)
            return
        
        # Check if there are datasets without segmentation
        datasets_without_seg = [d for d in self.data_manager.datasets if not d.has_segmentation]
        print(f"DEBUG:   Datasets without segmentation: {len(datasets_without_seg)}")
        
        if datasets_without_seg and not any(d.has_segmentation for d in self.data_manager.datasets):
            # All datasets are calculating
            item = QListWidgetItem(f'Calculating segmentation for {len(datasets_without_seg)} datasets...')
            item.setFlags(Qt.NoItemFlags)
            item.setForeground(Qt.blue)
            self.dataset_list.addItem(item)
            return
        
        # Display ALL datasets (already sorted by uncertainty desc, then size desc)
        print(f"DEBUG:   Displaying all {len(self.data_manager.datasets)} datasets:")
        
        for i, triplet in enumerate(self.data_manager.datasets):
            # Skip datasets without segmentation (they're being calculated)
            if not triplet.has_segmentation:
                print(f"DEBUG:     {i+1}. {triplet.base_name}: calculating...")
                continue
            
            uncertainty = triplet.max_uncertainty_excluding_noise if triplet.max_uncertainty_excluding_noise is not None else 0.0
            voxel_count = triplet.voxel_count_at_max_uncertainty if triplet.voxel_count_at_max_uncertainty is not None else 0
            
            temp_indicator = " [TEMP]" if triplet.has_temp else ""
            print(f"DEBUG:     {i+1}. {triplet.base_name}: uncertainty={uncertainty:.3f}, size={voxel_count}{temp_indicator}")
            
            # Create display text: "dataset_name (uncertainty: X.XXX, size: YYYY) [TEMP]"
            display_text = f"{triplet.base_name} (uncertainty: {uncertainty:.3f}, size: {voxel_count}){temp_indicator}"
            
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, triplet)  # Store triplet in item
            
            # Color code based on uncertainty (higher = higher priority)
            if uncertainty > 0.7:
                item.setForeground(Qt.red)  # High priority (high uncertainty)
            elif uncertainty > 0.4:
                item.setForeground(Qt.darkYellow)  # Medium priority
            # else: default color (low priority)
            
            if triplet.has_temp:
                # Make temp files bold
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            
            self.dataset_list.addItem(item)
    
    def _on_item_double_clicked(self, item: QListWidgetItem):
        """Handle double-click on dataset item"""
        self._load_item(item)
    
    def _load_selected(self):
        """Load the selected dataset"""
        current_item = self.dataset_list.currentItem()
        if current_item:
            self._load_item(current_item)
        else:
            QMessageBox.information(self, 'No Selection', 
                                   'Please select a dataset to load')
    
    def _load_item(self, item: QListWidgetItem):
        """Load a specific dataset item"""
        triplet = item.data(Qt.UserRole)
        if triplet:
            self.selected_triplet = triplet
            self.dataset_selected.emit(triplet)
            self.accept()
    
    def get_selected_triplet(self) -> Optional['DatasetTriplet']:
        """Get the selected dataset triplet"""
        return self.selected_triplet
