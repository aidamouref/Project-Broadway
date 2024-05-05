#FUNCIONES PROYECTO BROADWAY


#Función para hacer el encoding the la variable Theatre
def clean_theatre(x):
    if x in group_theatre:
        return "other"
    else:
        return x


#Funciones para transformar mis variables en Log o Raíz Cuadrada:
def log_transform_clean_(x):
    x=np.log(x)
    if np.isfinite(x) and x>0:
        return x
    else:
        return np.NAN 
    


def sqrt_transform_clean_(x):
    if np.isfinite(x) and x>=0:
        return np.sqrt(x)
    else:
        return np.NAN # We are returning NaNs so that we can replace them later


#Función para hacer el encoding the la variable Show
def musical(show):
    legit=['Light in the Piazza', 'A Gentleman’s Guide to Love and Murder', 'The Bridges of Madison County', 'Grey Gardens', 'Jane Eyre', 'Doctor Zhivago', 'It Shoulda Been You', 'A Tale of Two Cities', 'The Woman in White', 'Mary Poppins', 'Little Women', 'Dracula', 'Amour', 'The Visit', 'A Class Act', 'White Christmas', 'Nice Work if You Can Get It', 'Chaplin', 'An American in Paris', 'LoveMusik', 'The People in the Picture, The Scottsboro Boys', 'The Story of My Life','A Catered Affair', 'A Year with Frog and Toad', 'By Jeeves', 'Chitty Chitty Bang Bang', 'James Joyce’s The Dead', 'Allegiance', 'Amazing Grace', 'The Green Bird', 'Lestat', 'Anastasia', 'Amelie', 'Holiday Inn', 'War Paint', 'Flying Over Sunset']
    mixed=['The Producers', 'The Book of Mormon', 'Thoroughly Modern Millie', 'Avenue Q', 'Newsies', 'The 25th Annual Putnam County Spelling Bee', 'The Drowsy Chaperone', 'The Color Purple', 'Curtains', 'Shrek', 'Catch Me If You Can', 'The Full Monty', 'Caroline or Change', 'Dirty Rotten Scoundrels', 'Legally Blonde', 'Wicked', 'Matilda', 'Hairspray', 'Young Frankenstein', 'Big Fish', 'The Addams Family', 'Fun Home', 'Seussical', 'Ghost', 'Something Rotten', 'Spamalot', 'Sweet Smell of Success', 'The Wild Party', 'Sister Act', 'A Christmas Story', 'Billy Elliot', 'The Little Mermaid', 'Aladdin', 'The Pirate Queen', 'Violet', 'Xanadu', 'Bullets Over Broadway', 'Dear Evan Hansen', '9 to 5', 'Cry-Baby', 'Charlie and the Chocolate Factory', 'Scandalous', 'Women on the Verge of a Nervous Breakdown', 'Rocky', 'Elf', 'First Date', 'Urban Cowboy', '[title of show]', 'Urinetown', '13', 'Finding Neverland', 'If/Then', 'School of Rock', 'Honeymoon in Vegas', 'The Adventures of Tom Sawyer', 'Charlie and the Chocolate Factory', 'Paradise Square', 'Mr. Saturday Night', 'A Strange Loop', 'Kimberly Akimbo']
    pop_rock=['Spring Awakening', 'Hamilton', 'Next to Normal', 'Once', 'Bloody', 'Bloody Andrew Jackson', 'Brooklyn', 'Aida', 'Bring it On', 'Kinky Boots', 'Hedwig and the Angry Inch', 'In the Heights', 'Wonderland', 'The Last Ship', 'Spider-Man: Turn off the Dark', 'Lysistrata Jones', 'Thou Shalt Not', 'Leap of Faith', 'The Wedding Singer', 'Tarzan', 'High Fidelity', 'Bonnie and Clyde', 'Dogfight', 'Passing Strange', 'Hands on a Hardbody', 'Taboo', 'Glory Days', 'Waitress', 'American Psycho', 'Bombay Dreams', 'Natasha', 'Pierre & The Great Comet of 1812','School of Rock', 'Hadestown', 'King Kong', 'Diana: The Musical', 'Almost Famous', 'Be More Chill', 'The Lightning Thief', 'Six', 'Jersey Boys', 'All Shook Up', 'American Idiot', 'Beautiful: The Carol King Musical', 'Mamma Mia!', 'Million Dollar Quartet', 'Motown', 'Rock of Ages', 'Holler If Ya Hear Me', 'Priscilla – Queen of the Desert', 'Good Vibrations', 'A Night with Janis Joplin', 'Baby It’s You!', 'Ring of Fire', 'The Boy from Oz', 'Soul Doctor', 'Lennon', 'Rain', 'Hot Feet', 'Movin’ Out', 'On Your Feet!', 'The Look of Love', 'The Times They Are a-Changin’', 'Everyday Rapture', 'Disaster!', 'Escape to Margarita Ville', 'Summer: The Donna Summer Musical', 'Head Over Heels', 'Ain’t Too Proud', 'Moulin Rouge', 'The Cher Show', 'Jagged Little Pill', 'MJ: The Musical', '&Juliet']


    if show in legit:
        return "legit"
    elif show in mixed:
        return "mixed"
    elif show in pop_rock:
        return "pop_rock"
    else:
        return "other"