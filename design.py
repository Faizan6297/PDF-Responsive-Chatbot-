css = '''
<style>
.chat-message {
    padding: 1.2rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 15%;
}
.chat-message .avatar img {
  max-width: 48px;
  max-height: 48px;
  border-radius: 25%;
  object-fit: cover;
}
.chat-message .message {
  width: 40%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://th.bing.com/th?id=OIP.9O5_FyaQUETpjQv6SUaiNQHaLF&w=204&h=305&c=8&rs=1&qlt=90&o=6&dpr=1.3&pid=3.1&rm=2" style="max-height: 48px; max-width: 48px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://th.bing.com/th/id/OIP.NS9tjpxpg5T6zud95PcXlAHaHa?w=196&h=196&c=7&r=0&o=5&dpr=1.3&pid=1.7">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''