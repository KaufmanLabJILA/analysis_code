# Import smtplib for the actual sending function
import smtplib
import string

# Import the email modules we'll need
from email.mime.text import MIMEText

user = 'kaufmanlabJILA'
passwd = 'Sr88IsABoson'
smtp_server = 'smtp.gmail.com:587'
sender = 'kaufmanlabJILA@gmail.com'
receivers = 'aaron.young@jila.colorado.edu, kaufmanlabJILA@gmail.com'

msg = MIMEText('test text')
msg['Subject'] = 'ALERT: Lab sensor'+str()+'has exceeded safe range.'
msg['From'] = sender
msg['To'] = receivers

# msg = str('From: '+ send+ 'To: '+ receive+ 'Subject: TEST'+'test text'+'\r\n')

server = smtplib.SMTP(smtp_server)
server.starttls()
server.login(user, passwd)
server.send_message(msg)
server.quit()
