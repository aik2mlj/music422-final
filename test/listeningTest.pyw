#!/usr/bin/env python

"""
ITU-R BS.1116-based Impairment test administer tool

Made for MUSIC 422 listening tests (HW6)
-----------------------------------------------------------------------
© 2009-25 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------


created: February 2012
last modified: March 2020 (Marina Bosi)
    ListeningTest.pyw
    Modified on March 3, 2020
    Download and install wxpython (pip install wxpython)
    Updated the API on March 3rd as follows:
        import wx.adv
        line 112 OPEN -> FD_OPEN
        line 118 CHANGE_DIR -> FD_CHANGE_DIR
        line 218, 221, 374, 377 wx.SOUND -> wx.adv.SOUND
"""

import wx
import wx.adv
from wx.lib.intctrl import IntCtrl
import os
import random
from datetime import datetime
import sys
from pydub import AudioSegment
from pydub.playback import play

#########################################
# Globals
#########################################


INSTRUCTIONS = (
    "This is a test tool to evaluate the quality of a perceptual audio "
    "codec comparing the coded version of a file vs. "
    "the original file. Before starting the test, make sure "
    "you have the original uncompressed file (reference) and the "
    "encoded/decoded ones in the same folder.\n\n"
    "A trial is the evaluation of a coded "
    "file against the reference.\n\n"
    "To start the test you'll be asked to select the reference file; "
    "upon selection, the tool will prepare an ITU-R BS.1116-based "
    "test sequence to evaluate all the other WAV (*.wav|*.WAV) files in the "
    "same folder where the reference is stored.\n\n"
    "Once all the trials are finished you will be presented with "
    "the results of the ITU-R BS.1116-based test. A log file will "
    "also be generated summarizing the results. The log file will "
    "be automatically generated in the folder where the "
    "audio material is being stored."
)

MIN_GRADE = 1.0
MAX_GRADE = 5.0

CURRENT_TRIAL_MESSAGE = "Current trial"

TEST_TITLE = "ITU-R BS.1116-based"

APP_COLOR = "#DDDDDD"


#########################################
# GUI classes
#########################################


class ITUR(wx.Frame):
    """
    Main frame ("window") of the app
    """

    def __init__(self, *args, **kwargs):
        super(ITUR, self).__init__(*args, **kwargs)
        self.SetBackgroundColour(APP_COLOR)
        self.InitUI()
        self.session_manager = SessionManager(1, self)

    def InitUI(self):
        self.settings_panel = SettingsPanel(self)
        self.test_panel = TestPanel(self, style=wx.SIMPLE_BORDER)

        mainVBox = wx.BoxSizer(wx.VERTICAL)

        startTestButton = wx.Button(self, label="Start Test")
        startTestButton.Bind(wx.EVT_BUTTON, self.startClick)

        instructionsButton = wx.Button(self, label="Instructions")
        instructionsButton.Bind(wx.EVT_BUTTON, self.instructionsClick)

        middleButtonBar = wx.BoxSizer(wx.HORIZONTAL)
        middleButtonBar.Add(instructionsButton)
        middleButtonBar.AddSpacer(10)
        middleButtonBar.Add(startTestButton)

        mainVBox.AddSpacer(15)
        mainVBox.Add(self.settings_panel, flag=wx.ALIGN_CENTER)
        mainVBox.AddSpacer(15)
        mainVBox.Add(middleButtonBar, flag=wx.ALIGN_CENTER)
        mainVBox.AddSpacer(30)
        mainVBox.Add(self.test_panel, flag=wx.ALIGN_CENTER)

        resetButton = wx.Button(self, label="Reset All")
        resetButton.Bind(wx.EVT_BUTTON, self.resetClick)

        mainVBox.AddStretchSpacer()
        mainVBox.AddSpacer(15)
        mainVBox.Add(resetButton, flag=wx.ALIGN_CENTER)
        mainVBox.AddSpacer(15)

        self.SetTitle("%s Impairment test administer" % (TEST_TITLE))
        self.SetSizer(mainVBox)
        self.SetSize((500, 600))
        # self.Fit()
        self.Centre()
        self.Show(True)

    def resetClick(self, event=None):
        self.test_panel.clearTrials()
        self.session_manager.clear()

    def startClick(self, event=None):
        self.resetClick()
        # TODO: check validity of log file
        wildcard = "WAV files (*.wav)|*.wav"

        dlg = wx.FileDialog(
            self,
            message="Choose a file",
            defaultFile="",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_CHANGE_DIR,
        )
        if dlg.ShowModal() == wx.ID_OK:
            selected = dlg.GetPaths()[0]  # single file selection
            # this starts the test!
            self.session_manager.useFolder(selected)
            self.settings_panel.enableControls(False)
        dlg.Destroy()

    def instructionsClick(self, event=None):
        dlg = wx.MessageBox(INSTRUCTIONS, "Instructions", wx.OK | wx.ICON_INFORMATION)

    def getLogFileName(self):
        return self.settings_panel.getLogFileName()

    def autoPlayReference(self):
        return self.settings_panel.autoPlayReference()


class TestPanel(wx.Panel):
    """
    Presents the interface for a single ITU-R BS.1116-based session.

    Contains controls (widgets) to play back files and enter the grades for
    each file being tested
    """

    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.SetBackgroundColour(parent.GetBackgroundColour())
        self.numTrials = 0

        vbox = wx.BoxSizer(wx.VERTICAL)

        titleSizer = wx.BoxSizer(wx.HORIZONTAL)

        font = wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.NORMAL, wx.FONTWEIGHT_BOLD)
        testTitle = wx.StaticText(self, label=("%s Test" % (TEST_TITLE)))
        testTitle.SetFont(font)

        font.SetWeight(wx.FONTWEIGHT_NORMAL)
        self.sessionDesc = wx.StaticText(self, label="Session -/-")
        self.sessionDesc.SetFont(font)

        self.fgs = wx.FlexGridSizer(cols=4, hgap=8, vgap=8)
        fgs = self.fgs

        next = wx.Button(self, label="Next")
        next.Bind(wx.EVT_BUTTON, self.nextClick)

        self.writeHeader()

        vbox.Add(testTitle, flag=wx.ALIGN_CENTER)
        vbox.Add(self.sessionDesc, flag=wx.ALIGN_CENTER)
        vbox.Add(fgs, proportion=1, flag=wx.ALL, border=15)
        vbox.AddSpacer(5)
        vbox.Add(next, flag=wx.ALIGN_CENTER)
        vbox.AddSpacer(15)

        self.SetSizer(vbox)

    def addRow(self, numRows=1):
        for i in range(numRows):
            trial = wx.StaticText(self, label="")
            font = wx.Font(11, wx.TELETYPE, wx.NORMAL, wx.NORMAL)
            trial.SetFont(font)
            trialNo = wx.StaticText(self, label=("%d" % (self.numTrials + 1)))
            gradeA = wx.TextCtrl(self, value="")
            gradeB = wx.TextCtrl(self, value="")
            self.fgs.AddMany([(trialNo, wx.ALIGN_CENTER), (gradeA), (gradeB), (trial)])
            self.numTrials += 1
        self.Layout()
        self.parent.Layout()
        return (gradeA, gradeB, trial, trialNo)

    def clearTrials(self):
        self.fgs.Clear(delete_windows=True)
        self.numTrials = 0
        self.writeHeader()
        self.Layout()

    def writeHeader(self):
        buttonA = wx.Button(self, label="Play A")
        buttonB = wx.Button(self, label="Play B")
        buttonRef = wx.Button(self, label="Play Reference")

        buttonA.Bind(wx.EVT_BUTTON, self.playA)
        buttonB.Bind(wx.EVT_BUTTON, self.playB)
        buttonRef.Bind(wx.EVT_BUTTON, self.playRef)

        gradeA = wx.StaticText(self, label="Grade A")
        gradeB = wx.StaticText(self, label="Grade B")
        trialNo = wx.StaticText(self, label="Trial No")

        self.fgs.AddMany(
            [
                (wx.Frame(self)),
                (buttonA),
                (buttonB),
                (buttonRef),
                (trialNo),
                (gradeA),
                (gradeB),
                (wx.Frame(self)),
            ]
        )

    def playFile(self, option):
        if not self.parent.session_manager.currentTrial():
            return
        file_path = self.parent.session_manager.currentTrial().getFileName(option)
        sound = AudioSegment.from_file(file_path, format="wav")
        play(sound)
        # sound = wx.adv.Sound(self.parent.session_manager.currentTrial().getFileName(option))
        # sound.Stop()
        # sound.Play(wx.SOUND_SYNC)
        # sound.Play(wx.adv.SOUND_ASYNC)

    def playA(self, event=None):
        self.playFile("a")

    def playB(self, event=None):
        self.playFile("b")

    def playRef(self, event=None):
        self.playFile(self.parent.session_manager.currentTrial().original)

    def nextClick(self, event=None):
        self.parent.session_manager.nextTrial()

    def newSession(self, sessionNumber, totalSessions):
        self.sessionDesc.SetLabel("Session %s/%s" % (sessionNumber, totalSessions))


class SettingsPanel(wx.Panel):
    """
    Display general options of the test
    """

    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.SetBackgroundColour(parent.GetBackgroundColour())

        vbox = wx.BoxSizer(wx.VERTICAL)

        font = wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.NORMAL, wx.FONTWEIGHT_BOLD)
        title = wx.StaticText(self, label="Settings")
        title.SetFont(font)
        vbox.Add(title, flag=wx.ALIGN_CENTER)

        fgs = wx.FlexGridSizer(3, 2, 4, 15)

        numSessions = wx.StaticText(self, label="Number of sessions:")

        playRef = wx.StaticText(self, label="Play reference at the beginning of each Trial:")
        logFile = wx.StaticText(self, label="Subject name/number:")

        self.numSessionsCtrl = IntCtrl(self, value=1, min=1)
        self.numSessionsCtrl.Bind(wx.EVT_TEXT, self.updateNumSessions)
        self.numSessionsCtrl.Disable()
        self.logTextCtrl = wx.TextCtrl(self, value="john")

        self.chkRef = wx.CheckBox(self)
        self.chkRef.SetValue(True)

        fgs.AddMany(
            [
                (numSessions),
                (self.numSessionsCtrl, 1, wx.EXPAND),
                (logFile, 1, wx.EXPAND),
                (self.logTextCtrl, 1, wx.EXPAND),
                (playRef),
                (self.chkRef),
            ]
        )

        vbox.Add(fgs, flag=wx.ALIGN_CENTER, border=15)
        self.SetSizer(vbox)

    def updateNumSessions(self, event=None):
        ctrl = event.GetEventObject()
        self.parent.resetClick()
        numSessions = int(ctrl.GetValue())
        if numSessions:
            self.parent.session_manager = SessionManager(numSessions, self.parent)

    def getLogFileName(self):
        return self.logTextCtrl.GetValue()

    def autoPlayReference(self):
        return self.chkRef.GetValue()

        self.numSessionsCtrl.Disable()

    def enableControls(self, enable=True):
        if enable:
            self.numSessionsCtrl.Disable()
        else:
            self.numSessionsCtrl.Disable()


#########################################
# ITU-R BS.1116-based test classes
#########################################


class Trial(object):
    """
    A single trial: a sing test file being compared against a reference file.
    It contains logic to verify the evaluation and compute the SDG of the trial
    """

    def __init__(self, reference_file, test_file):
        self.original = ["a", "b"][random.choice([0, 1])]
        self.test_file = test_file
        self.reference_file = reference_file
        self.ctrlA = None
        self.ctrlB = None
        self.label = None
        self.labelNo = None
        self.completed = False

    def isCompleted(self):
        if self.completed:
            # TODO: lock the UI
            return True
        try:
            self.gradeA = float(self.ctrlA.GetValue())
            self.gradeB = float(self.ctrlB.GetValue())
        except ValueError:
            msg = (
                "Trial %s is not complete: one of the grades is not a number! "
                "Grades must be numbers in the [1.0, 5.0] range." % (self.labelNo.GetLabel())
            )
            dlg = wx.MessageBox(msg, "Error", wx.OK | wx.ICON_ERROR)
            return False

        self.completed = (MIN_GRADE <= self.gradeA <= MAX_GRADE) & (
            MIN_GRADE <= self.gradeB <= MAX_GRADE
        )

        if not self.completed:
            msg = "Trial %s is not complete: one grades is out of the [1.0, 5.0] range." % (
                self.labelNo.GetLabel()
            )
            dlg = wx.MessageBox(msg, "Error", wx.OK | wx.ICON_ERROR)

        return self.completed

    def getFileName(self, option):  # option == 'a' or 'b'
        if option.lower() == self.original:
            return self.reference_file
        else:
            return self.test_file

    def getReferenceName(self):
        return self.reference_file

    def getSDG(self):
        return (self.gradeB - self.gradeA) if self.original == "a" else (self.gradeA - self.gradeB)


class Session(object):
    """
    A single ITU-R BS.1116-based session: one or more test files being evaluated
    against a single reference file.
    Each test file will be tested in a "Trial" (see Trial class above).
    """

    def __init__(self, name):
        self.name = name
        self.trials = []
        self.current_trial = -1
        self.completed = False

    def addTrial(self, reference, test):
        self.trials.append(Trial(reference, test))

    def isCompleted(self):
        return self.completed

    def nextTrial(self, auto_play_reference=False):
        if self.current_trial >= 0 and not self.trials[self.current_trial].isCompleted():
            return
        self.current_trial += 1
        if self.current_trial < len(self.trials):
            if self.current_trial > 0:
                self.trials[self.current_trial - 1].label.SetBackgroundColour("Green")
                self.trials[self.current_trial - 1].label.SetLabel(" " * len(CURRENT_TRIAL_MESSAGE))
            self.trials[self.current_trial].label.SetBackgroundColour("Yellow")
            self.trials[self.current_trial].label.SetLabel(CURRENT_TRIAL_MESSAGE)
            if auto_play_reference:
                file_path = self.trials[self.current_trial].getReferenceName()
                sound = AudioSegment.from_file(file_path, format="wav")
                play(sound)
                # sound = wx.adv.Sound(self.trials[self.current_trial].getReferenceName())
                # sound.Stop()
                # sound.Play(wx.SOUND_SYNC)
                # sound.Play(wx.adv.SOUND_ASYNC)
        else:
            self.completed = True

    def currentTrial(self):
        return self.trials[self.current_trial]


class SessionManager(object):
    """
    Handles a set of ITU-R BS.1116-based Sessions (see Session class above).

    Given an input reference file, it constructs a set of sessions to test all the other WAV files
    in the same folder as the reference file.

    Once all sessions have been completed, it displays the overall results.
    """

    def __init__(self, num_sessions=1, parent=None):
        self.num_sessions = num_sessions
        self.parent = parent
        self.test_panel = parent.test_panel  # HACK! (need to decouple this)
        self.clear()

    def clear(self):
        self.sessions = []
        self.cur_indx = -1
        self.parent.settings_panel.enableControls()  # HACK! (need to decouple this)

    def addSession(self, session_number):
        session = Session(session_number)
        self.sessions.append(session)
        self.test_panel.newSession(
            session_number, self.num_sessions
        )  # HACK! (need to decouple this)
        random.shuffle(self.test_files)  # randomize the test files order
        for f in self.test_files:
            session.addTrial(self.reference, f)
        # setup the UI
        self.test_panel.clearTrials()
        for trial in session.trials:
            ctrls = self.test_panel.addRow()
            trial.ctrlA = ctrls[0]
            trial.ctrlB = ctrls[1]
            trial.label = ctrls[2]
            trial.labelNo = ctrls[3]
            # for testing purposes
            # trial.ctrlA.SetValue(trial.getFileName('a'))
            # trial.ctrlB.SetValue(trial.getFileName('b'))

    def useFolder(self, selected="./input.wav"):
        folder_name = os.path.dirname(selected)
        file_name = os.path.basename(selected)
        self.test_files = []
        for name in os.listdir(folder_name):
            if name[-4:].lower() == ".wav":
                if name == file_name:
                    self.reference = os.path.join(folder_name, name)
                else:
                    self.test_files.append(os.path.join(folder_name, name))
        self.nextSession()

    def currentIndex(self):
        return self.cur_indx

    def nextTrial(self):
        if self.cur_indx >= self.num_sessions or self.cur_indx == -1:
            return
        session = self.sessions[self.cur_indx]
        session.nextTrial(self.parent.autoPlayReference())  # HACK! (need to decouple this)
        if session.isCompleted():
            self.nextSession()

    def nextSession(self):
        self.cur_indx += 1
        if self.cur_indx < self.num_sessions:
            self.addSession(self.cur_indx + 1)
            dlg = wx.MessageBox(
                "You will now start a new %s session" % (TEST_TITLE),
                "Info",
                wx.OK | wx.ICON_INFORMATION,
            )
            session = self.sessions[self.cur_indx]
            session.nextTrial(self.parent.autoPlayReference())  # HACK! (need to decouple this)
        else:
            self.test_panel.clearTrials()  # HACK! (need to decouple this)
            self.test_panel.newSession("-", "-")  # HACK! (need to decouple this)
            self.summarize()

    def currentTrial(self):
        return self.sessions[self.cur_indx].currentTrial() if self.cur_indx > -1 else None

    def summarize(self):
        summary = {}
        for f in self.test_files:
            summary[f] = {}
        for session in self.sessions:
            for trial in session.trials:
                summary[trial.test_file][session.name] = trial.getSDG()

        output = "# %s test\n" % (TEST_TITLE)
        output += "# Administered on %s\n" % (datetime.today().ctime())
        output += "# Reference file: %s\n" % (self.reference)
        output += "# Subject name/number: %s\n" % (self.parent.getLogFileName())
        max_length = 0
        for test_file, grades in summary.items():
            max_length = max(max_length, len(os.path.basename(test_file)))

        output += "Test file".ljust(max_length) + ", "
        for session, grade in grades.items():
            output += ("SDG").rjust(12) + ", "
        output += "\n"

        for test_file, grades in summary.items():
            output += ("%s" % os.path.basename(test_file)).ljust(max_length) + ", "
            for session, grade in grades.items():
                output += ("%.2f" % grade).rjust(12) + ", "
            output += "\n"

        # write to log file (append if already exists)
        f = open(
            self.parent.getLogFileName() + "_" + os.path.basename(self.reference)[:-4] + ".csv",
            "a+",
        )  # HACK! (need to decouple this)
        f.write(output)
        f.close()

        output = "\n" + "=" * 30 + "\n"
        output += "%s test\n" % (TEST_TITLE)
        output += "Administered on %s\n" % (datetime.today().ctime())
        output += "Reference file: %s\n" % (self.reference)
        output += "-" * 10 + "\n"
        for test_file, grades in summary.items():
            output += "%s:\n" % (test_file)
            for session, grade in grades.items():
                output += "\tSDG = %.2f\n" % (grade)
        output += "=" * 30 + "\n"

        dlg = wx.MessageBox(output, "Results", wx.OK | wx.ICON_HAND)
        self.clear()


def main():
    """
    Run the wxPython app
    """
    ex = wx.App()
    ITUR(None)
    ex.MainLoop()


if __name__ == "__main__":
    """
    Entry Point
    """
    main()
